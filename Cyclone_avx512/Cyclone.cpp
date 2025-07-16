//g++ -std=c++17 -Ofast -ffast-math -funroll-loops -ftree-vectorize -fstrict-aliasing -fno-semantic-interposition -fvect-cost-model=unlimited -fno-trapping-math -fipa-ra -mavx512f -mavx512vl -mavx512bw -mavx512dq -fipa-modref -flto -fassociative-math -fopenmp -mavx2 -mbmi2 -madx -o Cyclone Cyclone.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp Point.cpp ripemd160_avx2.cpp p2pkh_decoder.cpp sha256_avx2.cpp ripemd160_avx512.cpp sha256_avx512.cpp

//The software is developed for solving Satoshi's puzzles; any use for illegal purposes is strictly prohibited. The author is not responsible for any actions taken by the user when using this software for unlawful activities.
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
// Adding program modules
#include "p2pkh_decoder.h"
#include "sha256_avx2.h"
#include "ripemd160_avx2.h"
#include "sha256_avx512.h"
#include "ripemd160_avx512.h"
#include "SECP256K1.h"
#include "Point.h"
#include "Int.h"
#include "IntGroup.h"

//------------------------------------------------------------------------------
// Batch size: ±256 public keys (512), hashed in groups of 16 (AVX512).
static constexpr int POINTS_BATCH_SIZE = 256;
static constexpr int HASH_BATCH_SIZE   = 16;

// Status output and progress saving frequency
static constexpr double statusIntervalSec = 5.0;
static constexpr double saveProgressIntervalSec = 300.0;

static int g_progressSaveCount = 0;
static std::vector<std::string> g_threadPrivateKeys;

//------------------------------------------------------------------------------
void saveProgressToFile(const std::string &progressStr)
{
    std::ofstream ofs("progress.txt", std::ios::app);
    if (ofs) {
        ofs << progressStr << "\n";
    } else {
        std::cerr << "Cannot open progress.txt for writing\n";
    }
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

//------------------------------------------------------------------------------
//Converts a HEX string into a large number (a vector of 64-bit words, little-endian).

std::vector<uint64_t> hexToBigNum(const std::string& hex) {
    std::vector<uint64_t> bigNum;
    const size_t len = hex.size();
    bigNum.reserve((len + 15) / 16);
    for (size_t i = 0; i < len; i += 16) {
        size_t start = (len >= 16 + i) ? len - 16 - i : 0;
        size_t partLen = (len >= 16 + i) ? 16 : (len - i);
        uint64_t value = std::stoull(hex.substr(start, partLen), nullptr, 16);
        bigNum.push_back(value);
    }
    return bigNum;
}

//Reverse conversion to a HEX string (with correct leading zeros within blocks).

std::string bigNumToHex(const std::vector<uint64_t>& num) {
    std::ostringstream oss;
    for (auto it = num.rbegin(); it != num.rend(); ++it) {
         if (it != num.rbegin())
            oss << std::setw(16) << std::setfill('0');
        oss << std::hex << *it;
    }
    return oss.str();
}

std::vector<uint64_t> singleElementVector(uint64_t val) {
    return { val };
}

std::vector<uint64_t> bigNumAdd(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    std::vector<uint64_t> sum;
    sum.reserve(std::max(a.size(), b.size()) + 1);
    uint64_t carry = 0;
    for (size_t i = 0, sz = std::max(a.size(), b.size()); i < sz; ++i) {
        uint64_t x = (i < a.size()) ? a[i] : 0ULL;
        uint64_t y = (i < b.size()) ? b[i] : 0ULL;
        __uint128_t s = ( __uint128_t )x + ( __uint128_t )y + carry;
        carry = (uint64_t)(s >> 64);
        sum.push_back((uint64_t)s);
    }
    if (carry) sum.push_back(carry);
    return sum;
}

std::vector<uint64_t> bigNumSubtract(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    std::vector<uint64_t> diff = a;
    uint64_t borrow = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        uint64_t subtrahend = b[i];
        if (diff[i] < subtrahend + borrow) {
            diff[i] = diff[i] + (~0ULL) - subtrahend - borrow + 1ULL; // eqv diff[i] = diff[i] - subtrahend - borrow
            borrow = 1ULL;
        } else {
            diff[i] -= (subtrahend + borrow);
            borrow = 0ULL;
        }
    }
    
    for (size_t i = b.size(); i < diff.size() && borrow; ++i) {
        if (diff[i] == 0ULL) {
            diff[i] = ~0ULL;
        } else {
            diff[i] -= 1ULL;
            borrow = 0ULL;
        }
    }
    // delete leading zeros
    while (!diff.empty() && diff.back() == 0ULL)
        diff.pop_back();
    return diff;
}


std::pair<std::vector<uint64_t>, uint64_t> bigNumDivide(const std::vector<uint64_t>& a, uint64_t divisor) {
    std::vector<uint64_t> quotient(a.size(), 0ULL);
    uint64_t remainder = 0ULL;
    for (int i = (int)a.size() - 1; i >= 0; --i) {
        __uint128_t temp = ((__uint128_t)remainder << 64) | a[i];
        uint64_t q = (uint64_t)(temp / divisor);
        uint64_t r = (uint64_t)(temp % divisor);
        quotient[i] = q;
        remainder   = r;
    }
    while (!quotient.empty() && quotient.back() == 0ULL)
        quotient.pop_back();
    return { quotient, remainder };
}

long double hexStrToLongDouble(const std::string &hex) {
    long double result = 0.0L;
    for (char c : hex) {
        result *= 16.0L;
        if (c >= '0' && c <= '9')
            result += (c - '0');
        else if (c >= 'a' && c <= 'f')
            result += (c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            result += (c - 'A' + 10);
    }
    return result;
}

//------------------------------------------------------------------------------
static inline std::string padHexTo64(const std::string &hex) {
    return (hex.size() >= 64) ? hex : std::string(64 - hex.size(), '0') + hex;
}
static inline Int hexToInt(const std::string &hex) {
    Int number;
    char buf[65] = {0};
    std::strncpy(buf, hex.c_str(), 64);
    number.SetBase16(buf);
    return number;
}
static inline std::string intToHex(const Int &value) {
    Int temp;
    temp.Set((Int*)&value);
    return temp.GetBase16();
}
static inline bool intGreater(const Int &a, const Int &b) {
    std::string ha = ((Int&)a).GetBase16();
    std::string hb = ((Int&)b).GetBase16();
    if (ha.size() != hb.size()) return (ha.size() > hb.size());
    return (ha > hb);
}
static inline bool isEven(const Int &number) {
    return ((Int&)number).IsEven();
}

static inline std::string intXToHex64(const Int &x) {
    Int temp;
    temp.Set((Int*)&x);
    std::string hex = temp.GetBase16();
    if (hex.size() < 64)
        hex.insert(0, 64 - hex.size(), '0');
    return hex;
}

static inline std::string pointToCompressedHex(const Point &point) {
    return (isEven(point.y) ? "02" : "03") + intXToHex64(point.x);
}
static inline void pointToCompressedBin(const Point &point, uint8_t outCompressed[33]) {
    outCompressed[0] = isEven(point.y) ? 0x02 : 0x03;
    Int temp;
    temp.Set((Int*)&point.x);
    for (int i = 0; i < 32; i++) {
        outCompressed[1 + i] = (uint8_t)temp.GetByte(31 - i);
    }
}

//------------------------------------------------------------------------------
inline void prepareShaBlock(const uint8_t* dataSrc, size_t dataLen, uint8_t* outBlock) {
    std::fill_n(outBlock, 64, 0);
    std::memcpy(outBlock, dataSrc, dataLen);
    outBlock[dataLen] = 0x80;
    const uint32_t bitLen = (uint32_t)(dataLen * 8);
    outBlock[60] = (uint8_t)((bitLen >> 24) & 0xFF);
    outBlock[61] = (uint8_t)((bitLen >> 16) & 0xFF);
    outBlock[62] = (uint8_t)((bitLen >>  8) & 0xFF);
    outBlock[63] = (uint8_t)( bitLen        & 0xFF);
}
inline void prepareRipemdBlock(const uint8_t* dataSrc, uint8_t* outBlock) {
    std::fill_n(outBlock, 64, 0);
    std::memcpy(outBlock, dataSrc, 32);
    outBlock[32] = 0x80;
    const uint32_t bitLen = 256;
    outBlock[60] = (uint8_t)((bitLen >> 24) & 0xFF);
    outBlock[61] = (uint8_t)((bitLen >> 16) & 0xFF);
    outBlock[62] = (uint8_t)((bitLen >>  8) & 0xFF);
    outBlock[63] = (uint8_t)( bitLen        & 0xFF);
}

// Computing hash160 using avx512 (16 hashes per try)
static void computeHash160BatchBinSingle(int numKeys,
                                         uint8_t pubKeys[][33],
                                         uint8_t hashResults[][20])
{
    std::array<std::array<uint8_t, 64>, HASH_BATCH_SIZE> shaInputs;
    std::array<std::array<uint8_t, 32>, HASH_BATCH_SIZE> shaOutputs;
    std::array<std::array<uint8_t, 64>, HASH_BATCH_SIZE> ripemdInputs;
    std::array<std::array<uint8_t, 20>, HASH_BATCH_SIZE> ripemdOutputs;

    const size_t totalBatches = (numKeys + (HASH_BATCH_SIZE - 1)) / HASH_BATCH_SIZE;
    for (size_t batch = 0; batch < totalBatches; batch++) {
        const size_t batchCount = std::min<size_t>(HASH_BATCH_SIZE, numKeys - batch * HASH_BATCH_SIZE);
        for (size_t i = 0; i < batchCount; i++) {
            const size_t idx = batch * HASH_BATCH_SIZE + i;
            prepareShaBlock(pubKeys[idx], 33, shaInputs[i].data());
        }
        for (size_t i = batchCount; i < HASH_BATCH_SIZE; i++) {
            std::memcpy(shaInputs[i].data(), shaInputs[0].data(), 64);
        }
        const uint8_t* inPtr[HASH_BATCH_SIZE];
        uint8_t* outPtr[HASH_BATCH_SIZE];
        for (int i = 0; i < HASH_BATCH_SIZE; i++) {
            inPtr[i]  = shaInputs[i].data();
            outPtr[i] = shaOutputs[i].data();
        }
        sha256_avx512_16B(inPtr[0], inPtr[1], inPtr[2], inPtr[3],
                      inPtr[4], inPtr[5], inPtr[6], inPtr[7],
                      inPtr[8], inPtr[9], inPtr[10], inPtr[11],
                      inPtr[12], inPtr[13], inPtr[14], inPtr[15],
                      outPtr[0], outPtr[1], outPtr[2], outPtr[3],
                      outPtr[4], outPtr[5], outPtr[6], outPtr[7],
                      outPtr[8], outPtr[9], outPtr[10], outPtr[11],
                      outPtr[12], outPtr[13], outPtr[14], outPtr[15]);

        for (size_t i = 0; i < batchCount; i++) {
            prepareRipemdBlock(shaOutputs[i].data(), ripemdInputs[i].data());
        }
        for (size_t i = batchCount; i < HASH_BATCH_SIZE; i++) {
            std::memcpy(ripemdInputs[i].data(), ripemdInputs[0].data(), 64);
        }
        for (int i = 0; i < HASH_BATCH_SIZE; i++) {
            inPtr[i]  = ripemdInputs[i].data();
            outPtr[i] = ripemdOutputs[i].data();
        }
        ripemd160avx512::ripemd160avx512_32(
            (unsigned char*)inPtr[0],
            (unsigned char*)inPtr[1],
            (unsigned char*)inPtr[2],
            (unsigned char*)inPtr[3],
            (unsigned char*)inPtr[4],
            (unsigned char*)inPtr[5],
            (unsigned char*)inPtr[6],
            (unsigned char*)inPtr[7],
            (unsigned char*)inPtr[8],
            (unsigned char*)inPtr[9],
            (unsigned char*)inPtr[10],
            (unsigned char*)inPtr[11],
            (unsigned char*)inPtr[12],
            (unsigned char*)inPtr[13],
            (unsigned char*)inPtr[14],
            (unsigned char*)inPtr[15],
            outPtr[0], outPtr[1], outPtr[2], outPtr[3],
            outPtr[4], outPtr[5], outPtr[6], outPtr[7],
            outPtr[8], outPtr[9], outPtr[10], outPtr[11],
            outPtr[12], outPtr[13], outPtr[14], outPtr[15]
        );
        for (size_t i = 0; i < batchCount; i++) {
            const size_t idx = batch * HASH_BATCH_SIZE + i;
            std::memcpy(hashResults[idx], ripemdOutputs[i].data(), 20);
        }
    }
}


//------------------------------------------------------------------------------
static void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " -a <Base58_P2PKH> -r <START:END>\n";
}

static std::string formatElapsedTime(double seconds) {
    int hrs = (int)seconds / 3600;
    int mins = ((int)seconds % 3600) / 60;
    int secs = (int)seconds % 60;
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << hrs << ":"
        << std::setw(2) << std::setfill('0') << mins << ":"
        << std::setw(2) << std::setfill('0') << secs;
    return oss.str();
}

//------------------------------------------------------------------------------
static void printStatsBlock(int numCPUs, const std::string &targetAddr,
                            const std::string &rangeStr, double mkeysPerSec,
                            unsigned long long totalChecked, double elapsedTime,
                            int progressSaves, long double progressPercent)
{
    static bool firstPrint = true;
    if (!firstPrint) {
        std::cout << "\033[9A";
    } else {
        firstPrint = false;
    }
    std::cout << "================= WORK IN PROGRESS =================\n";
    std::cout << "Target Address: " << targetAddr << "\n";
    std::cout << "CPU Threads   : " << numCPUs << "\n";
    std::cout << "Mkeys/s       : " << std::fixed << std::setprecision(2) << mkeysPerSec << "\n";
    std::cout << "Total Checked : " << totalChecked << "\n";
    std::cout << "Elapsed Time  : " << formatElapsedTime(elapsedTime) << "\n";
    std::cout << "Range         : " << rangeStr << "\n";
    std::cout << "Progress      : " << std::fixed << std::setprecision(4) << progressPercent << " %\n";
    std::cout << "Progress Save : " << progressSaves << "\n";
    std::cout.flush();
}

//------------------------------------------------------------------------------
struct ThreadRange {
    std::string startHex;
    std::string endHex;
};

static std::vector<ThreadRange> g_threadRanges;

//------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    bool addressProvided = false, rangeProvided = false;
    std::string targetAddress, rangeInput;
    std::vector<uint8_t> targetHash160;

    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-a") && i + 1 < argc) {
            targetAddress = argv[++i];
            addressProvided = true;
            try {
                targetHash160 = P2PKHDecoder::getHash160(targetAddress);
                if (targetHash160.size() != 20)
                    throw std::invalid_argument("Invalid hash160 length.");
            } catch (const std::exception &ex) {
                std::cerr << "Error parsing address: " << ex.what() << "\n";
                return 1;
            }
        } else if (!std::strcmp(argv[i], "-r") && i + 1 < argc) {
            rangeInput = argv[++i];
            rangeProvided = true;
        } else {
            std::cerr << "Unknown parameter: " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }
    if (!addressProvided || !rangeProvided) {
        std::cerr << "Both -a <Base58_P2PKH> and -r <START:END> are required!\n";
        printUsage(argv[0]);
        return 1;
    }

    const size_t colonPos = rangeInput.find(':');
    if (colonPos == std::string::npos) {
        std::cerr << "Invalid range format. Use <START:END> in HEX.\n";
        return 1;
    }
    const std::string rangeStartHex = rangeInput.substr(0, colonPos);
    const std::string rangeEndHex   = rangeInput.substr(colonPos + 1);

    auto rangeStart = hexToBigNum(rangeStartHex);
    auto rangeEnd   = hexToBigNum(rangeEndHex);

    bool validRange = false;
    if (rangeStart.size() < rangeEnd.size()) {
        validRange = true;
    } else if (rangeStart.size() > rangeEnd.size()) {
        validRange = false;
    } else {
        validRange = true;
        for (int i = (int)rangeStart.size() - 1; i >= 0; --i) {
            if (rangeStart[i] < rangeEnd[i]) {
                break;
            } else if (rangeStart[i] > rangeEnd[i]) {
                validRange = false;
                break;
            }
        }
    }
    if (!validRange) {
        std::cerr << "Range start must be <= range end.\n";
        return 1;
    }

    auto rangeSize = bigNumSubtract(rangeEnd, rangeStart);
    rangeSize = bigNumAdd(rangeSize, singleElementVector(1ULL));

    const std::string rangeSizeHex = bigNumToHex(rangeSize);
    
    const long double totalRangeLD = hexStrToLongDouble(rangeSizeHex);

    const int numCPUs = omp_get_num_procs();
    g_threadPrivateKeys.resize(numCPUs, "0");

    auto [chunkSize, remainder] = bigNumDivide(rangeSize, (uint64_t)numCPUs);
    g_threadRanges.resize(numCPUs);

    std::vector<uint64_t> currentStart = rangeStart;
    for (int t = 0; t < numCPUs; t++) {
        auto currentEnd = bigNumAdd(currentStart, chunkSize);
        if (t < (int)remainder) {
            currentEnd = bigNumAdd(currentEnd, singleElementVector(1ULL));
        }
        currentEnd = bigNumSubtract(currentEnd, singleElementVector(1ULL));

        g_threadRanges[t].startHex = bigNumToHex(currentStart);
        g_threadRanges[t].endHex   = bigNumToHex(currentEnd);

        currentStart = bigNumAdd(currentEnd, singleElementVector(1ULL));
    }
    const std::string displayRange = g_threadRanges.front().startHex + ":" + g_threadRanges.back().endHex;

    unsigned long long globalComparedCount = 0ULL;
    double globalElapsedTime = 0.0;
    double mkeysPerSec       = 0.0;

    const auto tStart = std::chrono::high_resolution_clock::now();
    auto lastStatusTime = tStart;
    auto lastSaveTime   = tStart;

    bool matchFound            = false;
    std::string foundPrivateKeyHex, foundPublicKeyHex, foundWIF;

    Secp256K1 secp;
    secp.Init();

    // PARRALEL COMPUTING BLOCK
    #pragma omp parallel num_threads(numCPUs) \
      shared(globalComparedCount, globalElapsedTime, mkeysPerSec, matchFound, \
             foundPrivateKeyHex, foundPublicKeyHex, foundWIF, \
             tStart, lastStatusTime, lastSaveTime, g_progressSaveCount, \
             g_threadPrivateKeys)
    {
        const int threadId = omp_get_thread_num();

        Int privateKey = hexToInt(g_threadRanges[threadId].startHex);
        const Int threadRangeEnd = hexToInt(g_threadRanges[threadId].endHex);

        #pragma omp critical
        {
            g_threadPrivateKeys[threadId] = padHexTo64(intToHex(privateKey));

        }

        // Precomputing +i*G and -i*G for i=0..255
        std::vector<Point> plusPoints(POINTS_BATCH_SIZE);
        std::vector<Point> minusPoints(POINTS_BATCH_SIZE);
        for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
            Int tmp; tmp.SetInt32(i);
            Point p = secp.ComputePublicKey(&tmp); 
            plusPoints[i] = p;
            p.y.ModNeg();
            minusPoints[i] = p;
        }

        // Arrays for batch-adding
        std::vector<Int>  deltaX(POINTS_BATCH_SIZE);
        IntGroup modGroup(POINTS_BATCH_SIZE);

        // Save 512 publickeys
        const int fullBatchSize = 2 * POINTS_BATCH_SIZE;
        std::vector<Point> pointBatch(fullBatchSize);

        // Buffers for hashing
        uint8_t localPubKeys[fullBatchSize][33];
        uint8_t localHashResults[HASH_BATCH_SIZE][20];
        int localBatchCount = 0;
        int pointIndices[HASH_BATCH_SIZE];

        // Local count
        unsigned long long localComparedCount = 0ULL;

        // Download the target (hash160) в __m128i for fast compare
        __m128i target16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(targetHash160.data()));

        // main
        while (true) {
            if (intGreater(privateKey, threadRangeEnd)) {
                break;
            }
            // startPoint = privateKey * G
            Int currentBatchKey;
            currentBatchKey.Set(&privateKey);
            Point startPoint = secp.ComputePublicKey(&currentBatchKey);

            #pragma omp critical
            {
                g_threadPrivateKeys[threadId] = padHexTo64(intToHex(privateKey));
            }

            // Divide the batch of 512 keys into 2 blocks of 256 keys, count +256 and -256 from the center G-point of the batch
            // First pointBatch[0..255] +
            for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
                deltaX[i].ModSub(&plusPoints[i].x, &startPoint.x);
            }
            modGroup.Set(deltaX.data());
            modGroup.ModInv();
            for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
                Point tempPoint = startPoint;
                Int deltaY;
                deltaY.ModSub(&plusPoints[i].y, &startPoint.y);
                Int slope;
                slope.ModMulK1(&deltaY, &deltaX[i]);
                Int slopeSq;
                slopeSq.ModSquareK1(&slope);

                Int tmpX;
                tmpX.Set(&startPoint.x);
                tmpX.ModNeg();
                tmpX.ModAdd(&slopeSq);
                tmpX.ModSub(&plusPoints[i].x);
                tempPoint.x.Set(&tmpX);

                Int diffX;
                diffX.Set(&startPoint.x);
                diffX.ModSub(&tempPoint.x);
                diffX.ModMulK1(&slope);
                tempPoint.y.ModNeg();
                tempPoint.y.ModAdd(&diffX);

                pointBatch[i] = tempPoint;
            }

            // Second pointBatch[256..511] -
            for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
                Point tempPoint = startPoint;
                Int deltaY;
                deltaY.ModSub(&minusPoints[i].y, &startPoint.y);
                Int slope;
                slope.ModMulK1(&deltaY, &deltaX[i]);
                Int slopeSq;
                slopeSq.ModSquareK1(&slope);

                Int tmpX;
                tmpX.Set(&startPoint.x);
                tmpX.ModNeg();
                tmpX.ModAdd(&slopeSq);
                tmpX.ModSub(&minusPoints[i].x);
                tempPoint.x.Set(&tmpX);

                Int diffX;
                diffX.Set(&startPoint.x);
                diffX.ModSub(&tempPoint.x);
                diffX.ModMulK1(&slope);
                tempPoint.y.ModNeg();
                tempPoint.y.ModAdd(&diffX);

                pointBatch[POINTS_BATCH_SIZE + i] = tempPoint;
            }

            // Construct local buffeer
            for (int i = 0; i < fullBatchSize; i++) {
                pointToCompressedBin(pointBatch[i], localPubKeys[localBatchCount]);
                pointIndices[localBatchCount] = i;
                localBatchCount++;

                // 8 keys are ready - time to use avx512
                if (localBatchCount == HASH_BATCH_SIZE) {
                    computeHash160BatchBinSingle(localBatchCount, localPubKeys, localHashResults);
                    // Results check
                    for (int j = 0; j < HASH_BATCH_SIZE; j++) {
                        __m128i cand16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(localHashResults[j]));
                        __m128i cmp = _mm_cmpeq_epi8(cand16, target16);
                        if (_mm_movemask_epi8(cmp) == 0xFFFF) {
                            // Checking last 4 bytes (20 - 16)
                            if (!matchFound && std::memcmp(localHashResults[j], targetHash160.data(), 20) == 0) {
                                #pragma omp critical
                                {
                                    if (!matchFound) {
                                        matchFound = true;
                                        auto tEndTime = std::chrono::high_resolution_clock::now();
                                        globalElapsedTime = std::chrono::duration<double>(tEndTime - tStart).count();
                                        mkeysPerSec = (double)(globalComparedCount + localComparedCount) / globalElapsedTime / 1e6;

                                        // Recovering private key
                                        Int matchingPrivateKey;
                                        matchingPrivateKey.Set(&currentBatchKey);
                                        int idx = pointIndices[j];
                                        if (idx < 256) {
                                            Int offset; offset.SetInt32(idx);
                                            matchingPrivateKey.Add(&offset);
                                        } else {
                                            Int offset; offset.SetInt32(idx - 256);
                                            matchingPrivateKey.Sub(&offset);
                                        }
                                        foundPrivateKeyHex = padHexTo64(intToHex(matchingPrivateKey));
                                        Point matchedPoint = pointBatch[idx];
                                        foundPublicKeyHex  = pointToCompressedHex(matchedPoint);
                                        foundWIF = P2PKHDecoder::compute_wif(foundPrivateKeyHex, true);
                                    }
                                }
                                #pragma omp cancel parallel
                            }
                            localComparedCount++;
                        } else {
                            localComparedCount++;
                        }
                    }
                    localBatchCount = 0;
                }
            }

            // Next step
            {
                Int step;
                step.SetInt32(fullBatchSize - 2); // 510
                privateKey.Add(&step);
            }

            // Time to show status
            auto now = std::chrono::high_resolution_clock::now();
            double secondsSinceStatus = std::chrono::duration<double>(now - lastStatusTime).count();
            if (secondsSinceStatus >= statusIntervalSec) {
                #pragma omp critical
                {
                    globalComparedCount += localComparedCount;
                    localComparedCount = 0ULL;
                    globalElapsedTime = std::chrono::duration<double>(now - tStart).count();
                    mkeysPerSec = (double)globalComparedCount / globalElapsedTime / 1e6;

                    long double progressPercent = 0.0L;
                    if (totalRangeLD > 0.0L) {
                        progressPercent = ((long double)globalComparedCount / totalRangeLD) * 100.0L;
                    }
                    printStatsBlock(numCPUs, targetAddress, displayRange,
                                    mkeysPerSec, globalComparedCount,
                                    globalElapsedTime, g_progressSaveCount,
                                    progressPercent);
                    lastStatusTime = now;
                }
            }

            // Save progress periodically
            auto nowSave = std::chrono::high_resolution_clock::now();
            double secondsSinceSave = std::chrono::duration<double>(nowSave - lastSaveTime).count();
            if (secondsSinceSave >= saveProgressIntervalSec) {
                #pragma omp critical
                {
                    if (threadId == 0) {
                        g_progressSaveCount++;
                        std::ostringstream oss;
                        oss << "Progress Save #" << g_progressSaveCount << " at "
                            << std::chrono::duration<double>(nowSave - tStart).count() << " sec: "
                            << "TotalChecked=" << globalComparedCount << ", "
                            << "ElapsedTime=" << formatElapsedTime(globalElapsedTime) << ", "
                            << "Mkeys/s=" << std::fixed << std::setprecision(2) << mkeysPerSec << "\n";
                        for (int k = 0; k < numCPUs; k++) {
                            oss << "Thread Key " << k << ": " << g_threadPrivateKeys[k] << "\n";
                        }
                        saveProgressToFile(oss.str());
                        lastSaveTime = nowSave;
                    }
                }
            }

            if (matchFound) {
                break;
            }
        } // while(true)

        // Adding local count
        #pragma omp atomic
        globalComparedCount += localComparedCount;
    } // end of parralel section

    // Main results
    auto tEnd = std::chrono::high_resolution_clock::now();
    globalElapsedTime = std::chrono::duration<double>(tEnd - tStart).count();

    if (!matchFound) {
        mkeysPerSec = (double)globalComparedCount / globalElapsedTime / 1e6;
        std::cout << "\nNo match found.\n";
        std::cout << "Total Checked : " << globalComparedCount << "\n";
        std::cout << "Elapsed Time  : " << formatElapsedTime(globalElapsedTime) << "\n";
        std::cout << "Speed         : " << mkeysPerSec << " Mkeys/s\n";
        return 0;
    }
    
    writeFoundKey(foundPrivateKeyHex, foundPublicKeyHex, foundWIF, targetAddress);  

    // If the key was found
    std::cout << "================== FOUND MATCH! ==================\n";
    std::cout << "Private Key   : " << foundPrivateKeyHex << "\n";
    std::cout << "Public Key    : " << foundPublicKeyHex << "\n";
    std::cout << "WIF           : " << foundWIF << "\n";
    std::cout << "P2PKH Address : " << targetAddress << "\n";
    std::cout << "Total Checked : " << globalComparedCount << "\n";
    std::cout << "Elapsed Time  : " << formatElapsedTime(globalElapsedTime) << "\n";
    std::cout << "Speed         : " << mkeysPerSec << " Mkeys/s\n";
    return 0;
}
