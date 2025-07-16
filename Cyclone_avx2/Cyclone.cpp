#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <chrono>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <omp.h>
#include <array>
#include <utility>
#include <cstdint>
#include <climits>

#include "p2pkh_decoder.h"
#include "sha256_avx2.h"
#include "ripemd160_avx2.h"
#include "SECP256K1.h"
#include "Point.h"
#include "Int.h"
#include "IntGroup.h"

static constexpr int    POINTS_BATCH_SIZE       = 256;
static constexpr int    HASH_BATCH_SIZE         = 8;
static constexpr double STATUS_INTERVAL_SEC     = 5.0;
static constexpr double SAVE_PROGRESS_INTERVAL  = 300.0;

static int                          g_progressSaveCount = 0;
static unsigned long long           g_candidatesFound   = 0ULL;
static unsigned long long           g_jumpsCount        = 0ULL;
static uint64_t                     g_jumpSize          = 0ULL;
static std::vector<std::string>     g_threadPrivateKeys;
static bool                         g_saveCandidates    = false;


static inline std::string bytesToHex(const uint8_t* data, size_t len)
{
    static constexpr char lut[] = "0123456789abcdef";
    std::string out; out.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        uint8_t b = data[i];
        out.push_back(lut[b >> 4]);
        out.push_back(lut[b & 0x0F]);
    }
    return out;
}

static void writeFoundKey(const std::string& privHex,
                          const std::string& pubHex,
                          const std::string& wif,
                          const std::string& address)
{
    std::ofstream ofs("found_keys.txt", std::ios::app);
    if (!ofs) {
        std::cerr << "Cannot open found_keys.txt for writing\n";
        return;
    }
    ofs << privHex << ' ' << pubHex << ' ' << wif << ' ' << address << '\n';
}


static void appendCandidateToFile(const std::string& privHex,
                                  const std::string& pubHex,
                                  const std::string& hash160Hex)
{
    ++g_candidatesFound;
    if (!g_saveCandidates) return;

#pragma omp critical(candidates_io)
    {
        std::ofstream ofs("candidates.txt", std::ios::app);
        if (ofs)
            ofs << privHex << ' ' << pubHex << ' ' << hash160Hex << '\n';
        else
            std::cerr << "Cannot open candidates.txt for writing\n";
    }
}

void saveProgressToFile(const std::string &progressStr)
{
    std::ofstream ofs("progress.txt", std::ios::app);
    if (ofs) ofs << progressStr << "\n";
    else     std::cerr << "Cannot open progress.txt for writing\n";
}

std::vector<uint64_t> hexToBigNum(const std::string& hex)
{
    std::vector<uint64_t> bigNum;
    const size_t len = hex.size();
    bigNum.reserve((len + 15) / 16);
    for (size_t i = 0; i < len; i += 16) {
        size_t start   = (len >= 16 + i) ? len - 16 - i : 0;
        size_t partLen = (len >= 16 + i) ? 16           : (len - i);
        uint64_t value = std::stoull(hex.substr(start, partLen), nullptr, 16);
        bigNum.push_back(value);
    }
    return bigNum;
}

std::string bigNumToHex(const std::vector<uint64_t>& num)
{
    std::ostringstream oss;
    oss << std::hex;
    for (auto it = num.rbegin(); it != num.rend(); ++it) {
        if (it != num.rbegin()) oss << std::setw(16) << std::setfill('0');
        oss << *it;
    }
    return oss.str();
}

std::vector<uint64_t> singleElementVector(uint64_t v) { return {v}; }

std::vector<uint64_t> bigNumAdd(const std::vector<uint64_t>& a,
                                const std::vector<uint64_t>& b)
{
    std::vector<uint64_t> s;
    s.reserve(std::max(a.size(), b.size()) + 1);
    uint64_t carry = 0;
    for (size_t i = 0, sz = std::max(a.size(), b.size()); i < sz; ++i) {
        uint64_t x = (i < a.size()) ? a[i] : 0ULL;
        uint64_t y = (i < b.size()) ? b[i] : 0ULL;
        __uint128_t t = (__uint128_t)x + y + carry;
        carry = uint64_t(t >> 64);
        s.push_back(uint64_t(t));
    }
    if (carry) s.push_back(carry);
    return s;
}

std::vector<uint64_t> bigNumSubtract(const std::vector<uint64_t>& a,
                                     const std::vector<uint64_t>& b)
{
    std::vector<uint64_t> d = a;
    uint64_t borrow = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        uint64_t sub = b[i];
        if (d[i] < sub + borrow) {
            d[i] = d[i] + (~0ULL) - sub - borrow + 1ULL;
            borrow = 1ULL;
        } else {
            d[i] -= sub + borrow;
            borrow = 0ULL;
        }
    }
    for (size_t i = b.size(); borrow && i < d.size(); ++i) {
        if (d[i] == 0ULL) d[i] = ~0ULL;
        else { d[i] -= 1ULL; borrow = 0ULL; }
    }
    while (!d.empty() && d.back() == 0ULL) d.pop_back();
    return d;
}

std::pair<std::vector<uint64_t>, uint64_t> bigNumDivide(
    const std::vector<uint64_t>& a, uint64_t divisor)
{
    std::vector<uint64_t> q(a.size(), 0ULL);
    uint64_t r = 0ULL;
    for (int i = int(a.size()) - 1; i >= 0; --i) {
        __uint128_t t = ((__uint128_t)r << 64) | a[i];
        q[i] = uint64_t(t / divisor);
        r    = uint64_t(t % divisor);
    }
    while (!q.empty() && q.back() == 0ULL) q.pop_back();
    return {q, r};
}

long double hexStrToLongDouble(const std::string& h)
{
    long double res = 0.0L;
    for (char c: h) {
        res *= 16.0L;
        if      (c >= '0' && c <= '9') res += (c - '0');
        else if (c >= 'a' && c <= 'f') res += (c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') res += (c - 'A' + 10);
    }
    return res;
}

static inline std::string padHexTo64(const std::string& h)
{
    return (h.size() >= 64) ? h : std::string(64 - h.size(), '0') + h;
}

static inline Int hexToInt(const std::string& h)
{
    Int n; char buf[65] = {0};
    std::strncpy(buf, h.c_str(), 64);
    n.SetBase16(buf);
    return n;
}

static inline std::string intToHex(const Int& v)
{
    Int t; t.Set((Int*)&v); return t.GetBase16();
}

static inline bool intGreater(const Int& a,const Int& b)
{
    std::string ha=((Int&)a).GetBase16(), hb=((Int&)b).GetBase16();
    return ha.size()!=hb.size() ? ha.size()>hb.size() : ha>hb;
}

static inline bool isEven(const Int& n) { return n.IsEven(); }

static inline std::string intXToHex64(const Int& x)
{
    Int t; t.Set((Int*)&x);
    std::string h=t.GetBase16();
    if (h.size()<64) h.insert(0,64-h.size(),'0');
    return h;
}

static inline std::string pointToCompressedHex(const Point& p)
{
    return (isEven(p.y) ? "02" : "03") + intXToHex64(p.x);
}

static inline void pointToCompressedBin(const Point& p,uint8_t out[33])
{
    out[0] = isEven(p.y) ? 0x02 : 0x03;
    Int t; t.Set((Int*)&p.x);
    for (int i = 0; i < 32; ++i)
        out[1+i] = uint8_t(t.GetByte(31-i));
}

inline void prepareShaBlock(const uint8_t* src,size_t len,uint8_t* out)
{
    std::fill_n(out,64,0);
    std::memcpy(out,src,len);
    out[len]=0x80;
    uint32_t bitLen = uint32_t(len*8);
    out[60]=uint8_t(bitLen>>24);
    out[61]=uint8_t(bitLen>>16);
    out[62]=uint8_t(bitLen>> 8);
    out[63]=uint8_t(bitLen    );
}

inline void prepareRipemdBlock(const uint8_t* src,uint8_t* out)
{
    std::fill_n(out,64,0);
    std::memcpy(out,src,32);
    out[32]=0x80;
    uint32_t bitLen=256;
    out[60]=uint8_t(bitLen>>24);
    out[61]=uint8_t(bitLen>>16);
    out[62]=uint8_t(bitLen>> 8);
    out[63]=uint8_t(bitLen    );
}

static inline bool isDeniedPub(const uint8_t pub[33], int denyHexLen)
{
    if (denyHexLen <= 0) return false;
    int fullBytes   = denyHexLen / 2;
    bool halfNibble = denyHexLen & 1;

    for (int i = 0; i < fullBytes; ++i)
        if (pub[1 + i] != 0x00) return false;          

    if (halfNibble) {
        if ((pub[1 + fullBytes] & 0xF0) != 0x00) return false; 
    }
    return true;   
}

static void computeHash160BatchBinSingle(int nKeys,
                                         uint8_t pub[][33],
                                         uint8_t outHash[][20])
{
    std::array<std::array<uint8_t,64>,HASH_BATCH_SIZE> shaIn;
    std::array<std::array<uint8_t,32>,HASH_BATCH_SIZE> shaOut;
    std::array<std::array<uint8_t,64>,HASH_BATCH_SIZE> ripIn;
    std::array<std::array<uint8_t,20>,HASH_BATCH_SIZE> ripOut;

    size_t nBatches=(nKeys+HASH_BATCH_SIZE-1)/HASH_BATCH_SIZE;

    for (size_t b = 0; b < nBatches; ++b) {
        size_t cnt = std::min<size_t>(HASH_BATCH_SIZE, nKeys - b*HASH_BATCH_SIZE);

        for (size_t i = 0; i < cnt; ++i)
            prepareShaBlock(pub[b*HASH_BATCH_SIZE+i],33,shaIn[i].data());
        for (size_t i = cnt; i < HASH_BATCH_SIZE; ++i)
            std::memcpy(shaIn[i].data(),shaIn[0].data(),64);

        const uint8_t* in[HASH_BATCH_SIZE];
        uint8_t*       out[HASH_BATCH_SIZE];
        for (int i = 0; i < HASH_BATCH_SIZE; ++i) {
            in[i]=shaIn[i].data();
            out[i]=shaOut[i].data();
        }
        sha256avx2_8B(in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7],
                      out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);

        for (size_t i = 0; i < cnt; ++i)
            prepareRipemdBlock(shaOut[i].data(),ripIn[i].data());
        for (size_t i = cnt; i < HASH_BATCH_SIZE; ++i)
            std::memcpy(ripIn[i].data(),ripIn[0].data(),64);

        for (int i = 0; i < HASH_BATCH_SIZE; ++i) {
            in[i]=ripIn[i].data();
            out[i]=ripOut[i].data();
        }
        ripemd160avx2::ripemd160avx2_32(
            (unsigned char*)in[0],(unsigned char*)in[1],(unsigned char*)in[2],
            (unsigned char*)in[3],(unsigned char*)in[4],(unsigned char*)in[5],
            (unsigned char*)in[6],(unsigned char*)in[7],
            out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);

        for (size_t i = 0; i < cnt; ++i)
            std::memcpy(outHash[b*HASH_BATCH_SIZE+i],ripOut[i].data(),20);
    }
}

static void printUsage(const char* prog)
{
    std::cerr<<"Usage: "<<prog
             <<" -a <Base58_P2PKH> -r <START:END>"
             <<" [-p <HEXLEN>] [-j <JUMP>] [-s]"
             <<" [-t <THREADS>] [--public-deny <HEXLEN>]\n";
}

static std::string formatElapsedTime(double sec)
{
    int h=int(sec)/3600, m=(int(sec)%3600)/60, s=int(sec)%60;
    std::ostringstream oss;
    oss<<std::setw(2)<<std::setfill('0')<<h<<":"
       <<std::setw(2)<<m<<":"
       <<std::setw(2)<<s;
    return oss.str();
}

static void printStats(int nCPU,
                       const std::string& addr,
                       const std::string& hashHex,
                       const std::string& range,
                       double mks,
                       unsigned long long checked,
                       double elapsed,
                       int saves,
                       long double prog,
                       bool showCand,
                       unsigned long long candCnt,
                       bool showJump,
                       unsigned long long jumpCnt)
{
    const int lines = 10 + (showCand?1:0) + (showJump?1:0);
    static bool first=true;
    if(!first) std::cout<<"\033["<<lines<<"A";
    else       first=false;

    std::cout<<"================= WORK IN PROGRESS =================\n"
             <<"Target Address: "<<addr<<"\n"
             <<"Hash160       : "<<hashHex<<"\n"
             <<"CPU Threads   : "<<nCPU<<"\n"
             <<"Mkeys/s       : "<<std::fixed<<std::setprecision(2)<<mks<<"\n"
             <<"Total Checked : "<<checked<<"\n"
             <<"Elapsed Time  : "<<formatElapsedTime(elapsed)<<"\n"
             <<"Range         : "<<range<<"\n"
             <<"Progress      : "<<std::fixed<<std::setprecision(4)<<prog<<" %\n"
             <<"Progress Save : "<<saves<<"\n";
    if (showCand) std::cout<<"Candidates    : "<<candCnt<<"\n";
    if (showJump) std::cout<<"Jumps         : "<<jumpCnt<<"\n";
    std::cout.flush();
}

struct ThreadRange { std::string startHex,endHex; };
static std::vector<ThreadRange> g_threadRanges;

int main(int argc, char* argv[])
{
    bool aOK=false, rOK=false, pOK=false, jOK=false, sOK=false;
    bool tOK=false,  denyOK=false;

    int  prefLenHex   = 0;
    uint64_t jumpSize = 0ULL;
    int  userThreads  = 0;     
    int  denyHexLen   = 0;   

    std::string targetAddress, rangeStr;
    std::vector<uint8_t> targetHash160;

    for(int i=1;i<argc;++i){
        if(!std::strcmp(argv[i],"-a") && i+1<argc){
            targetAddress=argv[++i]; aOK=true;
            targetHash160=P2PKHDecoder::getHash160(targetAddress);
        }
        else if(!std::strcmp(argv[i],"-r") && i+1<argc){
            rangeStr=argv[++i]; rOK=true;
        }
        else if(!std::strcmp(argv[i],"-p") && i+1<argc){
            prefLenHex=std::stoi(argv[++i]); pOK=true;
            if(prefLenHex<1||prefLenHex>40){
                std::cerr<<"-p must be 1-40\n"; return 1;
            }
        }
        else if(!std::strcmp(argv[i],"-j") && i+1<argc){
            jumpSize=std::stoull(argv[++i]); jOK=true;
            if(jumpSize==0){
                std::cerr<<"-j must be >0\n"; return 1;
            }
        }
        else if(!std::strcmp(argv[i],"-s")){
            sOK=true;
        }
        else if(!std::strcmp(argv[i],"-t") && i+1<argc){
            userThreads=std::stoi(argv[++i]); tOK=true;
            if(userThreads<1){
                std::cerr<<"-t must be >0\n"; return 1;
            }
        }
        else if(!std::strcmp(argv[i],"--public-deny") && i+1<argc){
            denyHexLen=std::stoi(argv[++i]); denyOK=true;
            if(denyHexLen<1||denyHexLen>64){
                std::cerr<<"--public-deny must be 1-64\n"; return 1;
            }
        }
        else{
            printUsage(argv[0]); return 1;
        }
    }
    if(!aOK||!rOK){ printUsage(argv[0]); return 1; }
    if(jOK&&!pOK){ std::cerr<<"-j requires -p\n"; return 1; }

    g_saveCandidates      = sOK;
    const bool partialEnabled  = pOK;
    const bool jumpEnabled     = jOK;
    const bool pubDenyEnabled  = denyOK;
    g_jumpSize                = jumpEnabled ? jumpSize : 0ULL;

    int hwThreads = omp_get_num_procs();
    int numCPUs   = tOK ? std::min(userThreads, hwThreads) : hwThreads;

    std::string targetHashHex = bytesToHex(targetHash160.data(),
                                           targetHash160.size());

    size_t colon=rangeStr.find(':');
    if(colon==std::string::npos){ std::cerr<<"Bad range\n"; return 1; }
    std::string startHex=rangeStr.substr(0,colon);
    std::string endHex  =rangeStr.substr(colon+1);

    auto startBN=hexToBigNum(startHex), endBN=hexToBigNum(endHex);

    bool okRange=false;
    if(startBN.size()<endBN.size()) okRange=true;
    else if(startBN.size()==endBN.size()){
        okRange=true;
        for(int i=int(startBN.size())-1;i>=0;--i){
            if(startBN[i]<endBN[i]) break;
            if(startBN[i]>endBN[i]){ okRange=false; break; }
        }
    }
    if(!okRange){ std::cerr<<"Range start > end\n"; return 1; }

    auto rangeSize=bigNumAdd(bigNumSubtract(endBN,startBN),
                             singleElementVector(1ULL));
    long double totalRangeLD=hexStrToLongDouble(bigNumToHex(rangeSize));

    g_threadPrivateKeys.assign(numCPUs,"0");

    auto [chunk,remainder]=bigNumDivide(rangeSize,(uint64_t)numCPUs);
    g_threadRanges.resize(numCPUs);
    std::vector<uint64_t> cur=startBN;
    for(int t=0;t<numCPUs;++t){
        auto e=bigNumAdd(cur,chunk);
        if(t<remainder) e=bigNumAdd(e,singleElementVector(1ULL));
        e=bigNumSubtract(e,singleElementVector(1ULL));
        g_threadRanges[t].startHex=bigNumToHex(cur);
        g_threadRanges[t].endHex  =bigNumToHex(e);
        cur=bigNumAdd(e,singleElementVector(1ULL));
    }
    std::string displayRange=g_threadRanges.front().startHex+":"+
                              g_threadRanges.back().endHex;

    unsigned long long globalChecked=0ULL;
    double             globalElapsed=0.0, mkeys=0.0;
    auto tStart   = std::chrono::high_resolution_clock::now();
    auto lastStat = tStart;
    auto lastSave = tStart;

    bool        matchFound=false;
    std::string foundPriv, foundPub, foundWIF;

    Secp256K1 secp; secp.Init();
    Int i512; i512.SetInt32(510);
    Point big512G=secp.ComputePublicKey(&i512);


#pragma omp parallel num_threads(numCPUs) \
    shared(globalChecked,globalElapsed,mkeys,matchFound, \
           foundPriv,foundPub,foundWIF, \
           tStart,lastStat,lastSave,g_progressSaveCount, \
           g_threadPrivateKeys,g_candidatesFound,g_jumpsCount)
    {
        int tid=omp_get_thread_num();

        Int   priv    = hexToInt(g_threadRanges[tid].startHex);
        const Int privEnd=hexToInt(g_threadRanges[tid].endHex);
        Point base=secp.ComputePublicKey(&priv);

        std::vector<Point> plus(POINTS_BATCH_SIZE), minus(POINTS_BATCH_SIZE);
        for(int i=0;i<POINTS_BATCH_SIZE;++i){
            Int t; t.SetInt32(i);
            Point p=secp.ComputePublicKey(&t);
            plus[i]=p; p.y.ModNeg(); minus[i]=p;
        }
        std::vector<Int>  deltaX(POINTS_BATCH_SIZE);
        IntGroup          modGrp(POINTS_BATCH_SIZE);

        const int fullBatch=2*POINTS_BATCH_SIZE;
        std::vector<Point> ptBatch(fullBatch);
        uint8_t pubKeys[HASH_BATCH_SIZE][33];
        uint8_t hashRes[HASH_BATCH_SIZE][20];
        int localCnt=0, idxArr[HASH_BATCH_SIZE];
        unsigned long long localChecked=0ULL;
        unsigned long long localJumps   =0ULL;

        Int jumpInt;
        if(jumpEnabled){
            std::ostringstream oss; oss << std::hex << g_jumpSize;
            jumpInt = hexToInt(oss.str());
        }

        while(!matchFound){
            if(intGreater(priv,privEnd)) break;

#pragma omp critical
            g_threadPrivateKeys[tid]=padHexTo64(intToHex(priv));

            for(int i=0;i<POINTS_BATCH_SIZE;++i){
                deltaX[i].ModSub(&plus[i].x,&base.x);
            }
            modGrp.Set(deltaX.data()); modGrp.ModInv();

            for(int i=0;i<POINTS_BATCH_SIZE;++i){
                Point r=base;
                Int dY; dY.ModSub(&plus[i].y,&base.y);
                Int k; k.ModMulK1(&dY,&deltaX[i]);
                Int k2; k2.ModSquareK1(&k);
                Int xNew; xNew.Set(&base.x); xNew.ModNeg(); xNew.ModAdd(&k2);
                xNew.ModSub(&plus[i].x); r.x.Set(&xNew);
                Int dx; dx.Set(&base.x); dx.ModSub(&r.x); dx.ModMulK1(&k);
                r.y.ModNeg(); r.y.ModAdd(&dx);
                ptBatch[i]=r;
            }
            for(int i=0;i<POINTS_BATCH_SIZE;++i){
                Point r=base;
                Int dY; dY.ModSub(&minus[i].y,&base.y);
                Int k; k.ModMulK1(&dY,&deltaX[i]);
                Int k2; k2.ModSquareK1(&k);
                Int xNew; xNew.Set(&base.x); xNew.ModNeg(); xNew.ModAdd(&k2);
                xNew.ModSub(&minus[i].x); r.x.Set(&xNew);
                Int dx; dx.Set(&base.x); dx.ModSub(&r.x); dx.ModMulK1(&k);
                r.y.ModNeg(); r.y.ModAdd(&dx);
                ptBatch[POINTS_BATCH_SIZE+i]=r;
            }

            unsigned int pendingJumps=0;

            for(int i=0;i<fullBatch;++i){
                uint8_t tmpPub[33];
                pointToCompressedBin(ptBatch[i], tmpPub);


                if(pubDenyEnabled && isDeniedPub(tmpPub, denyHexLen)){
                    ++localChecked;      
                    continue;             
                }

                std::memcpy(pubKeys[localCnt], tmpPub, 33);
                idxArr[localCnt]=i;
                ++localCnt;

                if(localCnt==HASH_BATCH_SIZE){
                    computeHash160BatchBinSingle(localCnt,pubKeys,hashRes);
                    for(int j=0;j<HASH_BATCH_SIZE;++j){
                        const uint8_t* cand=hashRes[j];

                        bool prefixOK=true;
                        if(partialEnabled){
                            const int prefBytes  = prefLenHex/2;
                            const bool halfNibble= (prefLenHex&1);

                            if (prefBytes &&
                                std::memcmp(cand,targetHash160.data(),prefBytes)!=0)
                                prefixOK=false;

                            if(prefixOK && halfNibble){
                                if ((cand[prefBytes] & 0xF0) !=
                                    (targetHash160[prefBytes] & 0xF0))
                                    prefixOK=false;
                            }
                            if(prefixOK){
                                Int cPriv=priv;
                                int idx=idxArr[j];
                                if(idx<256){ Int off; off.SetInt32(idx); cPriv.Add(&off); }
                                else       { Int off; off.SetInt32(idx-256); cPriv.Sub(&off); }

                                appendCandidateToFile(
                                    padHexTo64(intToHex(cPriv)),
                                    pointToCompressedHex(ptBatch[idx]),
                                    bytesToHex(cand,20)
                                );
                                if(jumpEnabled) ++pendingJumps;
                            }
                        }

                        if(std::memcmp(cand,targetHash160.data(),20)==0){
#pragma omp critical(full_match)
                            {
                                if(!matchFound){
                                    matchFound=true;
                                    Int mPriv=priv;
                                    int idx=idxArr[j];
                                    if(idx<256){ Int off; off.SetInt32(idx); mPriv.Add(&off); }
                                    else       { Int off; off.SetInt32(idx-256); mPriv.Sub(&off); }
                                    foundPriv=padHexTo64(intToHex(mPriv));
                                    foundPub=pointToCompressedHex(ptBatch[idx]);
                                    foundWIF=P2PKHDecoder::compute_wif(foundPriv,true);
                                }
                            }
#pragma omp cancel parallel
                        }
                        ++localChecked;
                    }
                    localCnt=0;
                }
            } 


            if(jumpEnabled && pendingJumps>0){
                for(unsigned int pj=0; pj<pendingJumps; ++pj)
                    priv.Add(&jumpInt);   

                base = secp.ComputePublicKey(&priv);

                unsigned long long skipped =
                    static_cast<unsigned long long>(pendingJumps) * g_jumpSize;
                localChecked += skipped;
                localJumps   += pendingJumps;

#pragma omp atomic
                g_jumpsCount += pendingJumps;

                pendingJumps  = 0;
                if(intGreater(priv,privEnd)) break;
            }

            {
                Int step; step.SetInt32(fullBatch-2);
                priv.Add(&step);
                base=secp.AddDirect(base,big512G);
            }

            auto now=std::chrono::high_resolution_clock::now();
            if(std::chrono::duration<double>(now-lastStat).count()
               >= STATUS_INTERVAL_SEC)
            {
#pragma omp critical
                {
                    globalChecked   += localChecked; localChecked=0ULL;
                    globalElapsed    =
                        std::chrono::duration<double>(now - tStart).count();
                    mkeys            = globalChecked/globalElapsed/1e6;
                    long double prog = totalRangeLD>0.0L
                        ? (globalChecked/totalRangeLD*100.0L)
                        : 0.0L;

                    printStats(numCPUs,targetAddress,targetHashHex,displayRange,
                               mkeys,globalChecked,globalElapsed,
                               g_progressSaveCount,prog,
                               partialEnabled,g_candidatesFound,
                               jumpEnabled,g_jumpsCount);
                    lastStat=now;
                }
            }

            if(std::chrono::duration<double>(now-lastSave).count()
               >= SAVE_PROGRESS_INTERVAL)
            {
#pragma omp critical
                if(tid==0){
                    g_progressSaveCount++;
                    auto nowSave=std::chrono::high_resolution_clock::now();
                    double sinceStart=
                        std::chrono::duration<double>(nowSave - tStart).count();

                    std::ostringstream oss;
                    oss<<"Progress Save #"<<g_progressSaveCount
                       <<" at "<<sinceStart<<" sec: "
                       <<"TotalChecked="<<globalChecked<<", "
                       <<"ElapsedTime="<<formatElapsedTime(globalElapsed)<<", "
                       <<"Mkeys/s="<<std::fixed<<std::setprecision(2)
                                  <<mkeys<<"\n";
                    for(int k=0;k<numCPUs;++k){
                        oss<<"Thread Key "<<k<<": "<<g_threadPrivateKeys[k]<<"\n";
                    }
                    saveProgressToFile(oss.str());
                    lastSave=now;
                }
            }
        } 

#pragma omp atomic
        globalChecked += localChecked;
    } 

    if(!matchFound){
        std::cout<<"\nNo match found.\n";
        return 0;
    }
    writeFoundKey(foundPriv, foundPub, foundWIF, targetAddress);

    std::cout<<"================== FOUND MATCH! ==================\n"
             <<"Private Key   : "<<foundPriv<<"\n"
             <<"Public Key    : "<<foundPub<<"\n"
             <<"WIF           : "<<foundWIF<<"\n"
             <<"P2PKH Address : "<<targetAddress<<"\n";
    return 0;
}
