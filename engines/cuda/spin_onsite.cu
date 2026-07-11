#include "spin_onsite.hpp"
#include "simulation_box_device.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace nep_adapters::cuda_backend {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kMaxSpinCompress = 4;
constexpr int kMaxSpinBasis = 8;
constexpr int kSpinDeg2Count = 6;
constexpr int kSpinDeg3Count = 10;
constexpr int kSpinDeg4Count = 15;
constexpr int kSpinPrimitiveCount = 128;
constexpr int kSpinPrimitiveSlots = 88;
constexpr int kSpinChiralQohCount = 300;

__device__ __constant__ unsigned short kSpinChiralQohPacked[kSpinChiralQohCount] = {
    20, 23, 29, 35, 37, 46, 52, 55, 61, 67, 69, 78,
    86, 88, 92, 99, 101, 110, 118, 120, 124, 132, 135, 141,
    145, 146, 153, 154, 278, 280, 284, 288, 289, 290, 297, 298,
    299, 310, 312, 316, 320, 321, 322, 329, 330, 331, 340, 343,
    349, 352, 353, 354, 361, 362, 363, 372, 375, 381, 390, 392,
    396, 403, 405, 414, 528, 529, 530, 537, 538, 539, 550, 552,
    556, 560, 561, 562, 569, 570, 571, 582, 584, 588, 595, 597,
    606, 614, 616, 620, 627, 629, 638, 640, 641, 642, 649, 650,
    651, 660, 663, 669, 772, 775, 781, 800, 801, 802, 809, 810,
    811, 822, 824, 828, 832, 833, 834, 841, 842, 843, 852, 855,
    861, 864, 865, 866, 873, 874, 875, 884, 887, 893, 902, 904,
    908, 915, 917, 926, 1030, 1032, 1036, 1059, 1061, 1070, 1076, 1079,
    1085, 1091, 1093, 1102, 1110, 1112, 1116, 1123, 1125, 1134, 1142, 1144,
    1148, 1156, 1159, 1165, 1168, 1170, 1177, 1179, 1280, 1281, 1282, 1289,
    1290, 1291, 1316, 1319, 1325, 1331, 1333, 1342, 1348, 1351, 1357, 1360,
    1361, 1362, 1369, 1370, 1371, 1380, 1383, 1389, 1392, 1393, 1394, 1401,
    1402, 1403, 1411, 1413, 1422, 1430, 1432, 1436, 1539, 1541, 1550, 1552,
    1553, 1554, 1561, 1562, 1563, 1584, 1585, 1586, 1593, 1594, 1595, 1606,
    1608, 1612, 1619, 1621, 1630, 1638, 1640, 1644, 1651, 1653, 1662, 1664,
    1665, 1666, 1673, 1674, 1675, 1684, 1687, 1693, 1792, 1793, 1794, 1801,
    1802, 1803, 1811, 1813, 1822, 1843, 1845, 1854, 1860, 1863, 1869, 1872,
    1873, 1874, 1881, 1882, 1883, 1892, 1895, 1901, 1904, 1905, 1906, 1913,
    1914, 1915, 1923, 1925, 1934, 1942, 1944, 1948, 2054, 2056, 2060, 2068,
    2071, 2077, 2100, 2103, 2109, 2115, 2117, 2126, 2134, 2136, 2140, 2147,
    2149, 2158, 2166, 2168, 2172, 2180, 2183, 2189, 2192, 2193, 2202, 2203,
};

__device__ __constant__ double kSpinChiralQohCoeff[kSpinChiralQohCount] = {
    -1.0 / 7.0, -1.0 / 7.0, 6.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0, -6.0 / 7.0,
    4.0 / 7.0, -3.0 / 7.0, -3.0 / 7.0, -4.0 / 7.0, 3.0 / 7.0, 3.0 / 7.0,
    -2.0 / 7.0, -2.0 / 7.0, 12.0 / 7.0, 1.0 / 7.0, -6.0 / 7.0, 15.0 / 7.0,
    2.0 / 7.0, 2.0 / 7.0, -12.0 / 7.0, -1.0 / 7.0, 6.0 / 7.0, -15.0 / 7.0,
    2.0 / 7.0, -2.0 / 7.0, -12.0 / 7.0, 12.0 / 7.0, 4.0 / 7.0, -3.0 / 7.0,
    -3.0 / 7.0, -1.0 / 35.0, 4.0 / 35.0, 4.0 / 35.0, 3.0 / 35.0, 3.0 / 35.0,
    -27.0 / 35.0, -1.0 / 7.0, -1.0 / 7.0, 6.0 / 7.0, 4.0 / 35.0, 4.0 / 35.0,
    -1.0 / 35.0, -27.0 / 35.0, 3.0 / 35.0, 3.0 / 35.0, -2.0 / 7.0, -2.0 / 7.0,
    12.0 / 7.0, -1.0 / 35.0, -16.0 / 35.0, -11.0 / 35.0, 18.0 / 35.0, -12.0 / 35.0,
    78.0 / 35.0, 2.0 / 7.0, 2.0 / 7.0, -12.0 / 7.0, -11.0 / 7.0, 10.0 / 7.0,
    3.0 / 7.0, 4.0 / 7.0, -10.0 / 7.0, 18.0 / 7.0, 1.0 / 35.0, -4.0 / 35.0,
    -4.0 / 35.0, -3.0 / 35.0, -3.0 / 35.0, 27.0 / 35.0, 3.0 / 7.0, -4.0 / 7.0,
    3.0 / 7.0, -4.0 / 35.0, 1.0 / 35.0, -4.0 / 35.0, -3.0 / 35.0, 27.0 / 35.0,
    -3.0 / 35.0, 1.0 / 7.0, 1.0 / 7.0, -6.0 / 7.0, -2.0 / 7.0, -2.0 / 7.0,
    12.0 / 7.0, -10.0 / 7.0, 11.0 / 7.0, -3.0 / 7.0, 2.0 / 7.0, 2.0 / 7.0,
    -12.0 / 7.0, 1.0 / 35.0, 11.0 / 35.0, 16.0 / 35.0, 12.0 / 35.0, -18.0 / 35.0,
    -78.0 / 35.0, -4.0 / 7.0, 10.0 / 7.0, -18.0 / 7.0, -4.0 / 7.0, 3.0 / 7.0,
    3.0 / 7.0, -4.0 / 35.0, 1.0 / 35.0, -4.0 / 35.0, -3.0 / 35.0, 27.0 / 35.0,
    -3.0 / 35.0, 2.0 / 7.0, 2.0 / 7.0, -12.0 / 7.0, 16.0 / 35.0, 1.0 / 35.0,
    11.0 / 35.0, -18.0 / 35.0, -78.0 / 35.0, 12.0 / 35.0, 1.0 / 7.0, 1.0 / 7.0,
    -6.0 / 7.0, -4.0 / 35.0, -4.0 / 35.0, 1.0 / 35.0, 27.0 / 35.0, -3.0 / 35.0,
    -3.0 / 35.0, 11.0 / 7.0, -10.0 / 7.0, -3.0 / 7.0, -2.0 / 7.0, -2.0 / 7.0,
    12.0 / 7.0, 10.0 / 7.0, -4.0 / 7.0, -18.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0,
    -6.0 / 7.0, -1.0 / 7.0, -1.0 / 7.0, 6.0 / 7.0, 2.0 / 7.0, 2.0 / 7.0,
    -12.0 / 7.0, 6.0 / 7.0, -1.0 / 7.0, -15.0 / 7.0, -4.0 / 7.0, 3.0 / 7.0,
    3.0 / 7.0, -3.0 / 7.0, 4.0 / 7.0, -3.0 / 7.0, 1.0 / 7.0, -6.0 / 7.0,
    15.0 / 7.0, -2.0 / 7.0, -2.0 / 7.0, 12.0 / 7.0, -2.0 / 7.0, 2.0 / 7.0,
    12.0 / 7.0, -12.0 / 7.0, 4.0 / 35.0, -1.0 / 35.0, 4.0 / 35.0, 3.0 / 35.0,
    -27.0 / 35.0, 3.0 / 35.0, -3.0 / 7.0, 4.0 / 7.0, -3.0 / 7.0, 2.0 / 7.0,
    2.0 / 7.0, -12.0 / 7.0, 10.0 / 7.0, -11.0 / 7.0, 3.0 / 7.0, -1.0 / 35.0,
    4.0 / 35.0, 4.0 / 35.0, 3.0 / 35.0, 3.0 / 35.0, -27.0 / 35.0, -1.0 / 7.0,
    -1.0 / 7.0, 6.0 / 7.0, -11.0 / 35.0, -1.0 / 35.0, -16.0 / 35.0, -12.0 / 35.0,
    78.0 / 35.0, 18.0 / 35.0, -2.0 / 7.0, -2.0 / 7.0, 12.0 / 7.0, 4.0 / 7.0,
    -10.0 / 7.0, 18.0 / 7.0, 4.0 / 7.0, -3.0 / 7.0, -3.0 / 7.0, 4.0 / 35.0,
    4.0 / 35.0, -1.0 / 35.0, -27.0 / 35.0, 3.0 / 35.0, 3.0 / 35.0, -16.0 / 35.0,
    -11.0 / 35.0, -1.0 / 35.0, 78.0 / 35.0, 18.0 / 35.0, -12.0 / 35.0, -2.0 / 7.0,
    -2.0 / 7.0, 12.0 / 7.0, -11.0 / 7.0, 10.0 / 7.0, 3.0 / 7.0, 2.0 / 7.0,
    2.0 / 7.0, -12.0 / 7.0, -1.0 / 7.0, -1.0 / 7.0, 6.0 / 7.0, 4.0 / 35.0,
    -1.0 / 35.0, 4.0 / 35.0, 3.0 / 35.0, -27.0 / 35.0, 3.0 / 35.0, -10.0 / 7.0,
    4.0 / 7.0, 18.0 / 7.0, -4.0 / 35.0, -4.0 / 35.0, 1.0 / 35.0, 27.0 / 35.0,
    -3.0 / 35.0, -3.0 / 35.0, 3.0 / 7.0, -4.0 / 7.0, 3.0 / 7.0, -10.0 / 7.0,
    11.0 / 7.0, -3.0 / 7.0, -2.0 / 7.0, -2.0 / 7.0, 12.0 / 7.0, 11.0 / 35.0,
    16.0 / 35.0, 1.0 / 35.0, -78.0 / 35.0, 12.0 / 35.0, -18.0 / 35.0, 2.0 / 7.0,
    2.0 / 7.0, -12.0 / 7.0, 1.0 / 35.0, -4.0 / 35.0, -4.0 / 35.0, -3.0 / 35.0,
    -3.0 / 35.0, 27.0 / 35.0, 1.0 / 7.0, 1.0 / 7.0, -6.0 / 7.0, 10.0 / 7.0,
    -4.0 / 7.0, -18.0 / 7.0, -1.0 / 7.0, -1.0 / 7.0, 6.0 / 7.0, 1.0 / 7.0,
    1.0 / 7.0, -6.0 / 7.0, -6.0 / 7.0, 1.0 / 7.0, 15.0 / 7.0, -2.0 / 7.0,
    -2.0 / 7.0, 12.0 / 7.0, 6.0 / 7.0, -1.0 / 7.0, -15.0 / 7.0, 2.0 / 7.0,
    2.0 / 7.0, -12.0 / 7.0, -3.0 / 7.0, 4.0 / 7.0, -3.0 / 7.0, 3.0 / 7.0,
    -4.0 / 7.0, 3.0 / 7.0, 2.0 / 7.0, -2.0 / 7.0, -12.0 / 7.0, 12.0 / 7.0,
};

// The c4/l4 fast path contracts symmetric rank-2 moments. Merge Q_ab and
// Q_ba coefficients once so the hot Q-O-H loops do not repeat equivalent terms.
// The retained representatives are xx, xy, xz, yy, yz, zz; off-diagonal
// coefficients are C_xy + C_yx, C_xz + C_zx, and C_yz + C_zy.
constexpr int kSpinChiralQohSymCount = 192;
__device__ __constant__ unsigned short
kSpinChiralQohSymPacked[kSpinChiralQohSymCount] = {
    20, 23, 29, 35, 37, 46, 52, 55, 61, 67, 69, 78,
    86, 88, 92, 99, 101, 110, 118, 120, 124, 132, 135, 141,
    145, 146, 153, 154, 260, 263, 269, 278, 280, 284, 288, 289,
    298, 299, 310, 312, 316, 320, 321, 322, 329, 330, 331, 340,
    343, 349, 352, 353, 354, 361, 362, 363, 372, 375, 381, 390,
    392, 396, 403, 405, 515, 517, 526, 528, 530, 537, 539, 550,
    552, 556, 560, 561, 562, 569, 570, 571, 582, 584, 588, 595,
    597, 606, 614, 616, 620, 627, 629, 638, 640, 641, 642, 649,
    650, 651, 660, 663, 1030, 1032, 1036, 1059, 1061, 1070, 1076, 1079,
    1085, 1091, 1093, 1102, 1110, 1112, 1116, 1123, 1125, 1134, 1142, 1144,
    1148, 1156, 1159, 1165, 1168, 1170, 1177, 1179, 1281, 1282, 1289, 1290,
    1299, 1301, 1310, 1316, 1319, 1325, 1331, 1333, 1342, 1348, 1351, 1357,
    1360, 1361, 1362, 1369, 1370, 1371, 1380, 1383, 1389, 1392, 1393, 1394,
    1401, 1402, 1403, 1411, 1413, 1422, 1430, 1432, 2054, 2056, 2060, 2068,
    2071, 2077, 2100, 2103, 2109, 2115, 2117, 2126, 2134, 2136, 2140, 2147,
    2149, 2158, 2166, 2168, 2172, 2180, 2183, 2189, 2192, 2193, 2202, 2203,
};
__device__ __constant__ float
kSpinChiralQohSymCoeff[kSpinChiralQohSymCount] = {
    -1.0f / 7.0f, -1.0f / 7.0f, 6.0f / 7.0f, 1.0f / 7.0f, 1.0f / 7.0f, -6.0f / 7.0f,
    4.0f / 7.0f, -3.0f / 7.0f, -3.0f / 7.0f, -4.0f / 7.0f, 3.0f / 7.0f, 3.0f / 7.0f,
    -2.0f / 7.0f, -2.0f / 7.0f, 12.0f / 7.0f, 1.0f / 7.0f, -6.0f / 7.0f, 15.0f / 7.0f,
    2.0f / 7.0f, 2.0f / 7.0f, -12.0f / 7.0f, -1.0f / 7.0f, 6.0f / 7.0f, -15.0f / 7.0f,
    2.0f / 7.0f, -2.0f / 7.0f, -12.0f / 7.0f, 12.0f / 7.0f, -4.0f / 7.0f, 3.0f / 7.0f,
    3.0f / 7.0f, 4.0f / 7.0f, -3.0f / 7.0f, -3.0f / 7.0f, -1.0f / 7.0f, 1.0f / 7.0f,
    6.0f / 7.0f, -6.0f / 7.0f, 1.0f / 7.0f, 1.0f / 7.0f, -6.0f / 7.0f, 4.0f / 7.0f,
    1.0f / 7.0f, 2.0f / 7.0f, -9.0f / 7.0f, -15.0f / 7.0f, 3.0f / 7.0f, -1.0f / 7.0f,
    -1.0f / 7.0f, 6.0f / 7.0f, -1.0f / 7.0f, -4.0f / 7.0f, -2.0f / 7.0f, 9.0f / 7.0f,
    -3.0f / 7.0f, 15.0f / 7.0f, 13.0f / 7.0f, -8.0f / 7.0f, -15.0f / 7.0f, -13.0f / 7.0f,
    8.0f / 7.0f, 15.0f / 7.0f, 2.0f, -2.0f, 4.0f / 7.0f, -3.0f / 7.0f,
    -3.0f / 7.0f, 1.0f / 7.0f, -1.0f / 7.0f, -6.0f / 7.0f, 6.0f / 7.0f, 3.0f / 7.0f,
    -4.0f / 7.0f, 3.0f / 7.0f, -4.0f / 7.0f, -2.0f / 7.0f, -1.0f / 7.0f, 15.0f / 7.0f,
    9.0f / 7.0f, -3.0f / 7.0f, -1.0f / 7.0f, -1.0f / 7.0f, 6.0f / 7.0f, -13.0f / 7.0f,
    8.0f / 7.0f, 15.0f / 7.0f, -8.0f / 7.0f, 13.0f / 7.0f, -15.0f / 7.0f, 1.0f / 7.0f,
    1.0f / 7.0f, -6.0f / 7.0f, 1.0f / 7.0f, 2.0f / 7.0f, 4.0f / 7.0f, 3.0f / 7.0f,
    -9.0f / 7.0f, -15.0f / 7.0f, -2.0f, 2.0f, 1.0f / 7.0f, 1.0f / 7.0f,
    -6.0f / 7.0f, -1.0f / 7.0f, -1.0f / 7.0f, 6.0f / 7.0f, 2.0f / 7.0f, 2.0f / 7.0f,
    -12.0f / 7.0f, 6.0f / 7.0f, -1.0f / 7.0f, -15.0f / 7.0f, -4.0f / 7.0f, 3.0f / 7.0f,
    3.0f / 7.0f, -3.0f / 7.0f, 4.0f / 7.0f, -3.0f / 7.0f, 1.0f / 7.0f, -6.0f / 7.0f,
    15.0f / 7.0f, -2.0f / 7.0f, -2.0f / 7.0f, 12.0f / 7.0f, -2.0f / 7.0f, 2.0f / 7.0f,
    12.0f / 7.0f, -12.0f / 7.0f, -1.0f / 7.0f, 1.0f / 7.0f, 6.0f / 7.0f, -6.0f / 7.0f,
    3.0f / 7.0f, -4.0f / 7.0f, 3.0f / 7.0f, -3.0f / 7.0f, 4.0f / 7.0f, -3.0f / 7.0f,
    -8.0f / 7.0f, 13.0f / 7.0f, -15.0f / 7.0f, 8.0f / 7.0f, -13.0f / 7.0f, 15.0f / 7.0f,
    2.0f / 7.0f, 4.0f / 7.0f, 1.0f / 7.0f, -15.0f / 7.0f, 3.0f / 7.0f, -9.0f / 7.0f,
    1.0f / 7.0f, 1.0f / 7.0f, -6.0f / 7.0f, -2.0f / 7.0f, -1.0f / 7.0f, -4.0f / 7.0f,
    -3.0f / 7.0f, 15.0f / 7.0f, 9.0f / 7.0f, -1.0f / 7.0f, -1.0f / 7.0f, 6.0f / 7.0f,
    2.0f, -2.0f, -1.0f / 7.0f, -1.0f / 7.0f, 6.0f / 7.0f, 1.0f / 7.0f,
    1.0f / 7.0f, -6.0f / 7.0f, -6.0f / 7.0f, 1.0f / 7.0f, 15.0f / 7.0f, -2.0f / 7.0f,
    -2.0f / 7.0f, 12.0f / 7.0f, 6.0f / 7.0f, -1.0f / 7.0f, -15.0f / 7.0f, 2.0f / 7.0f,
    2.0f / 7.0f, -12.0f / 7.0f, -3.0f / 7.0f, 4.0f / 7.0f, -3.0f / 7.0f, 3.0f / 7.0f,
    -4.0f / 7.0f, 3.0f / 7.0f, 2.0f / 7.0f, -2.0f / 7.0f, -12.0f / 7.0f, 12.0f / 7.0f,
};

void check_cuda(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(message) + ": " + cudaGetErrorString(status));
  }
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool all_active(const std::vector<int>& mask, int num_types) {
  if (mask.empty()) {
    return true;
  }
  if (static_cast<int>(mask.size()) != num_types) {
    return false;
  }
  for (int value : mask) {
    if (value == 0) {
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ int idx2(int c, int k, int width) {
  return c * width + k;
}

template <bool AtomMajor, int ComponentCount>
__device__ __forceinline__ int spin_component_cache_index(
    int atom_stride,
    int atom,
    int component) {
  static_assert(ComponentCount > 0, "spin cache component count must be positive");
  if constexpr (AtomMajor) {
    return atom * ComponentCount + component;
  }
  return atom + atom_stride * component;
}

template <bool AtomMajor>
__device__ __forceinline__ int spin_component_cache_index(
    int atom_stride,
    int component_count,
    int atom,
    int component) {
  if constexpr (AtomMajor) {
    return atom * component_count + component;
  }
  return atom + atom_stride * component;
}

template <bool AtomMajor>
__device__ __forceinline__ int spin_component_cache_stride(int atom_stride) {
  if constexpr (AtomMajor) {
    return 1;
  }
  return atom_stride;
}

__device__ __forceinline__ int tensor3(int a, int b, int c) {
  return (a * 3 + b) * 3 + c;
}

__device__ __forceinline__ int tensor4(int a, int b, int c, int d) {
  return ((a * 3 + b) * 3 + c) * 3 + d;
}

__device__ __forceinline__ void cross3(
    const double* a,
    const double* b,
    double* out) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ __forceinline__ double dot3(const double* a, const double* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ __forceinline__ void stf_outer3(
    const double* a,
    const double* b,
    double* out) {
  const double trace = dot3(a, b) / 3.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double value = 0.5 * (a[i] * b[j] + a[j] * b[i]);
      if (i == j) {
        value -= trace;
      }
      out[3 * i + j] = value;
    }
  }
}

__device__ __forceinline__ int virial_internal_component(int row_major) {
  return row_major == 0 ? 0 :
         row_major == 1 ? 3 :
         row_major == 2 ? 4 :
         row_major == 3 ? 6 :
         row_major == 4 ? 1 :
         row_major == 5 ? 5 :
         row_major == 6 ? 7 :
         row_major == 7 ? 8 : 2;
}

__device__ __forceinline__ int spin_edge_cache_index(
    int atom_stride,
    int slot,
    int atom,
    int c) {
  return ((slot * atom_stride + atom) * 4) + c;
}

__device__ __forceinline__ void load_spin_edge_weight_cache(
    int atom_stride,
    int slot,
    int atom,
    const float* __restrict__ spin_edge_weights,
    double* weights) {
  for (int c = 0; c < 4; ++c) {
    weights[c] = static_cast<double>(
        spin_edge_weights[spin_edge_cache_index(atom_stride, slot, atom, c)]);
  }
}

__device__ __forceinline__ void load_spin_edge_weight_derivative_cache(
    int atom_stride,
    int slot,
    int atom,
    const float* __restrict__ spin_edge_weights,
    const float* __restrict__ spin_edge_weight_derivatives,
    double* weights,
    double* weight_derivatives) {
  for (int c = 0; c < 4; ++c) {
    const int index = spin_edge_cache_index(atom_stride, slot, atom, c);
    weights[c] = static_cast<double>(spin_edge_weights[index]);
    weight_derivatives[c] =
        static_cast<double>(spin_edge_weight_derivatives[index]);
  }
}

__device__ __forceinline__ int levi_civita(int a, int b, int c) {
  if (a == b || b == c || a == c) {
    return 0;
  }
  return ((a == 0 && b == 1 && c == 2) ||
          (a == 1 && b == 2 && c == 0) ||
          (a == 2 && b == 0 && c == 1)) ? 1 : -1;
}

__device__ __forceinline__ void find_fc_and_fcp(
    double rc,
    double rcinv,
    double r,
    double& fc,
    double& fcp) {
  if (r < rc) {
    const double x = r * rcinv;
    fc = 0.5 * cos(kPi * x) + 0.5;
    fcp = -0.5 * kPi * sin(kPi * x) * rcinv;
  } else {
    fc = 0.0;
    fcp = 0.0;
  }
}

__device__ void find_fn_and_fnp(
    int basis_size,
    double rcinv,
    double r,
    double fc,
    double fcp,
    double* fn,
    double* fnp) {
  fn[0] = fc;
  fnp[0] = fcp;
  if (basis_size == 0) {
    return;
  }
  const double r_scaled = r * rcinv;
  const double x = 2.0 * (r_scaled - 1.0) * (r_scaled - 1.0) - 1.0;
  fn[1] = 0.5 * (x + 1.0);
  fnp[1] = 2.0 * (r_scaled - 1.0) * rcinv * fc + fn[1] * fcp;
  fn[1] *= fc;
  double t0 = 1.0;
  double t1 = x;
  double u0 = 1.0;
  double u1 = 2.0 * x;
  for (int n = 2; n <= basis_size; ++n) {
    const double t2 = 2.0 * x * t1 - t0;
    fn[n] = 0.5 * (t2 + 1.0);
    fnp[n] = n * u1 * 2.0 * (r_scaled - 1.0) * rcinv;
    fnp[n] = fnp[n] * fc + fn[n] * fcp;
    fn[n] *= fc;
    const double u2 = 2.0 * x * u1 - u0;
    t0 = t1;
    t1 = t2;
    u0 = u1;
    u1 = u2;
  }
}

__device__ int real_spherical_harmonics_spin(
    const double* rhat,
    int ell,
    double* out) {
  const double x = rhat[0];
  const double y = rhat[1];
  const double z = rhat[2];
  if (ell == 2) {
    out[0] = sqrt(15.0 / (4.0 * kPi)) * x * y;
    out[1] = sqrt(15.0 / (4.0 * kPi)) * y * z;
    out[2] = sqrt(5.0 / (16.0 * kPi)) * (2.0 * z * z - x * x - y * y);
    out[3] = sqrt(15.0 / (4.0 * kPi)) * x * z;
    out[4] = sqrt(15.0 / (16.0 * kPi)) * (x * x - y * y);
    return 5;
  }
  if (ell == 3) {
    const double rho2 = x * x + y * y;
    out[0] = sqrt(35.0 / (32.0 * kPi)) * y * (3.0 * x * x - y * y);
    out[1] = sqrt(105.0 / (4.0 * kPi)) * x * y * z;
    out[2] = sqrt(21.0 / (32.0 * kPi)) * y * (4.0 * z * z - rho2);
    out[3] = sqrt(7.0 / (16.0 * kPi)) * z * (2.0 * z * z - 3.0 * rho2);
    out[4] = sqrt(21.0 / (32.0 * kPi)) * x * (4.0 * z * z - rho2);
    out[5] = sqrt(105.0 / (16.0 * kPi)) * z * (x * x - y * y);
    out[6] = sqrt(35.0 / (32.0 * kPi)) * x * (x * x - 3.0 * y * y);
    return 7;
  }
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  out[0] = 0.75 * sqrt(35.0 / kPi) * x * y * (x2 - y2);
  out[1] = 0.75 * sqrt(35.0 / (2.0 * kPi)) * y * z * (3.0 * x2 - y2);
  out[2] = 0.75 * sqrt(5.0 / kPi) * x * y * (7.0 * z2 - 1.0);
  out[3] = 0.75 * sqrt(5.0 / (2.0 * kPi)) * y * z * (7.0 * z2 - 3.0);
  out[4] = (3.0 / 16.0) * sqrt(1.0 / kPi) * (35.0 * z2 * z2 - 30.0 * z2 + 3.0);
  out[5] = 0.75 * sqrt(5.0 / (2.0 * kPi)) * x * z * (7.0 * z2 - 3.0);
  out[6] = 0.375 * sqrt(5.0 / kPi) * (x2 - y2) * (7.0 * z2 - 1.0);
  out[7] = 0.75 * sqrt(35.0 / (2.0 * kPi)) * x * z * (x2 - 3.0 * y2);
  out[8] = (3.0 / 16.0) * sqrt(35.0 / kPi) * (x2 * x2 - 6.0 * x2 * y2 + y2 * y2);
  return 9;
}

__device__ void add_real_spherical_harmonics_gradient(
    const double* r,
    int ell,
    const double* grad_y,
    double* grad_r) {
  const double x = r[0];
  const double y = r[1];
  const double z = r[2];
  if (ell == 2) {
    const double a = sqrt(15.0 / (4.0 * kPi));
    const double b = sqrt(5.0 / (16.0 * kPi));
    const double c = sqrt(15.0 / (16.0 * kPi));
    grad_r[0] += grad_y[0] * a * y - grad_y[2] * 2.0 * b * x +
                 grad_y[3] * a * z + grad_y[4] * 2.0 * c * x;
    grad_r[1] += grad_y[0] * a * x + grad_y[1] * a * z -
                 grad_y[2] * 2.0 * b * y - grad_y[4] * 2.0 * c * y;
    grad_r[2] += grad_y[1] * a * y + grad_y[2] * 4.0 * b * z +
                 grad_y[3] * a * x;
    return;
  }
  if (ell == 3) {
    const double x2 = x * x;
    const double y2 = y * y;
    const double z2 = z * z;
    const double rho2 = x2 + y2;
    const double a = sqrt(35.0 / (32.0 * kPi));
    const double b = sqrt(105.0 / (4.0 * kPi));
    const double c = sqrt(21.0 / (32.0 * kPi));
    const double d = sqrt(7.0 / (16.0 * kPi));
    const double e = sqrt(105.0 / (16.0 * kPi));
    grad_r[0] += grad_y[0] * 6.0 * a * x * y + grad_y[1] * b * y * z -
                 grad_y[2] * 2.0 * c * x * y - grad_y[3] * 6.0 * d * x * z +
                 grad_y[4] * c * (4.0 * z2 - 3.0 * x2 - y2) +
                 grad_y[5] * 2.0 * e * x * z + grad_y[6] * 3.0 * a * (x2 - y2);
    grad_r[1] += grad_y[0] * 3.0 * a * (x2 - y2) + grad_y[1] * b * x * z +
                 grad_y[2] * c * (4.0 * z2 - x2 - 3.0 * y2) -
                 grad_y[3] * 6.0 * d * y * z - grad_y[4] * 2.0 * c * x * y -
                 grad_y[5] * 2.0 * e * y * z - grad_y[6] * 6.0 * a * x * y;
    grad_r[2] += grad_y[1] * b * x * y + grad_y[2] * 8.0 * c * y * z +
                 grad_y[3] * d * (6.0 * z2 - 3.0 * rho2) +
                 grad_y[4] * 8.0 * c * x * z + grad_y[5] * e * (x2 - y2);
    return;
  }
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  const double a = 0.75 * sqrt(35.0 / kPi);
  const double b = 0.75 * sqrt(35.0 / (2.0 * kPi));
  const double c = 0.75 * sqrt(5.0 / kPi);
  const double d = 0.75 * sqrt(5.0 / (2.0 * kPi));
  const double e = (3.0 / 16.0) * sqrt(1.0 / kPi);
  const double f = 0.375 * sqrt(5.0 / kPi);
  const double g = (3.0 / 16.0) * sqrt(35.0 / kPi);
  grad_r[0] += grad_y[0] * a * y * (3.0 * x2 - y2) +
               grad_y[1] * b * 6.0 * x * y * z +
               grad_y[2] * c * y * (7.0 * z2 - 1.0) +
               grad_y[5] * d * z * (7.0 * z2 - 3.0) +
               grad_y[6] * 2.0 * f * x * (7.0 * z2 - 1.0) +
               grad_y[7] * b * z * (3.0 * x2 - 3.0 * y2) +
               grad_y[8] * g * (4.0 * x * x2 - 12.0 * x * y2);
  grad_r[1] += grad_y[0] * a * x * (x2 - 3.0 * y2) +
               grad_y[1] * b * z * (3.0 * x2 - 3.0 * y2) +
               grad_y[2] * c * x * (7.0 * z2 - 1.0) +
               grad_y[3] * d * z * (7.0 * z2 - 3.0) -
               grad_y[6] * 2.0 * f * y * (7.0 * z2 - 1.0) -
               grad_y[7] * b * 6.0 * x * y * z +
               grad_y[8] * g * (-12.0 * x2 * y + 4.0 * y * y2);
  grad_r[2] += grad_y[1] * b * y * (3.0 * x2 - y2) +
               grad_y[2] * c * 14.0 * x * y * z +
               grad_y[3] * d * y * (21.0 * z2 - 3.0) +
               grad_y[4] * e * (140.0 * z2 * z - 60.0 * z) +
               grad_y[5] * d * x * (21.0 * z2 - 3.0) +
               grad_y[6] * f * 14.0 * z * (x2 - y2) +
               grad_y[7] * b * x * (x2 - 3.0 * y2);
}

__device__ void fill_spin_monomials(const double* u, double* m3, double* m4) {
  const double x = u[0];
  const double y = u[1];
  const double z = u[2];
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  const double x3 = x2 * x;
  const double y3 = y2 * y;
  const double z3 = z2 * z;
  m3[0] = x3;
  m3[1] = y3;
  m3[2] = z3;
  m3[3] = x2 * y;
  m3[4] = x2 * z;
  m3[5] = x * y2;
  m3[6] = y2 * z;
  m3[7] = x * z2;
  m3[8] = y * z2;
  m3[9] = x * y * z;
  m4[0] = x2 * x2;
  m4[1] = y2 * y2;
  m4[2] = z2 * z2;
  m4[3] = x3 * y;
  m4[4] = x3 * z;
  m4[5] = x * y3;
  m4[6] = y3 * z;
  m4[7] = x * z3;
  m4[8] = y * z3;
  m4[9] = x2 * y2;
  m4[10] = x2 * z2;
  m4[11] = y2 * z2;
  m4[12] = x2 * y * z;
  m4[13] = x * y2 * z;
  m4[14] = x * y * z2;
}

__device__ __forceinline__ int spin_monomial_index(
    int degree,
    int nx,
    int ny,
    int nz) {
  if (degree == 2) {
    const int exp2[kSpinDeg2Count][3] = {
        {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
        {1, 1, 0}, {1, 0, 1}, {0, 1, 1}};
    for (int k = 0; k < kSpinDeg2Count; ++k) {
      if (exp2[k][0] == nx && exp2[k][1] == ny && exp2[k][2] == nz) {
        return k;
      }
    }
  } else if (degree == 3) {
    const int exp3[kSpinDeg3Count][3] = {
        {3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {2, 1, 0}, {2, 0, 1},
        {1, 2, 0}, {0, 2, 1}, {1, 0, 2}, {0, 1, 2}, {1, 1, 1}};
    for (int k = 0; k < kSpinDeg3Count; ++k) {
      if (exp3[k][0] == nx && exp3[k][1] == ny && exp3[k][2] == nz) {
        return k;
      }
    }
  } else {
    const int exp4[kSpinDeg4Count][3] = {
        {4, 0, 0}, {0, 4, 0}, {0, 0, 4}, {3, 1, 0}, {3, 0, 1},
        {1, 3, 0}, {0, 3, 1}, {1, 0, 3}, {0, 1, 3}, {2, 2, 0},
        {2, 0, 2}, {0, 2, 2}, {2, 1, 1}, {1, 2, 1}, {1, 1, 2}};
    for (int k = 0; k < kSpinDeg4Count; ++k) {
      if (exp4[k][0] == nx && exp4[k][1] == ny && exp4[k][2] == nz) {
        return k;
      }
    }
  }
  return 0;
}

__device__ double packed_value(const double* packed, int degree, const int* counts) {
  return packed[spin_monomial_index(degree, counts[0], counts[1], counts[2])];
}

__device__ void fill_spin_monomials2(const double* u, double* m2) {
  const double x = u[0];
  const double y = u[1];
  const double z = u[2];
  m2[0] = x * x;
  m2[1] = y * y;
  m2[2] = z * z;
  m2[3] = x * y;
  m2[4] = x * z;
  m2[5] = y * z;
}

__device__ double dot_spin_terms(const double* lhs, const double* rhs, int count) {
  double out = 0.0;
  for (int k = 0; k < count; ++k) {
    out += lhs[k] * rhs[k];
  }
  return out;
}

__device__ void unpack_rank3_spin_stf(const double* raw, double* out) {
  double trace[3] = {0.0, 0.0, 0.0};
  for (int c = 0; c < 3; ++c) {
    for (int e = 0; e < 3; ++e) {
      int counts[3] = {0, 0, 0};
      counts[e] += 2;
      ++counts[c];
      trace[c] += packed_value(raw, 3, counts);
    }
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        int counts[3] = {0, 0, 0};
        ++counts[a];
        ++counts[b];
        ++counts[c];
        const double traced =
            ((a == b) ? trace[c] : 0.0) +
            ((a == c) ? trace[b] : 0.0) +
            ((b == c) ? trace[a] : 0.0);
        out[tensor3(a, b, c)] = packed_value(raw, 3, counts) - traced / 5.0;
      }
    }
  }
}

__device__ void unpack_rank4_spin_stf(const double* raw, double* out) {
  double trace[9] = {0.0};
  for (int c = 0; c < 3; ++c) {
    for (int d = 0; d < 3; ++d) {
      for (int e = 0; e < 3; ++e) {
        int counts[3] = {0, 0, 0};
        counts[e] += 2;
        ++counts[c];
        ++counts[d];
        trace[3 * c + d] += packed_value(raw, 4, counts);
      }
    }
  }
  double double_trace = 0.0;
  for (int e = 0; e < 3; ++e) {
    double_trace += trace[3 * e + e];
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          int counts[3] = {0, 0, 0};
          ++counts[a];
          ++counts[b];
          ++counts[c];
          ++counts[d];
          const double six =
              ((a == b) ? trace[3 * c + d] : 0.0) +
              ((a == c) ? trace[3 * b + d] : 0.0) +
              ((a == d) ? trace[3 * b + c] : 0.0) +
              ((b == c) ? trace[3 * a + d] : 0.0) +
              ((b == d) ? trace[3 * a + c] : 0.0) +
              ((c == d) ? trace[3 * a + b] : 0.0);
          const double three =
              ((a == b && c == d) ? double_trace : 0.0) +
              ((a == c && b == d) ? double_trace : 0.0) +
              ((a == d && b == c) ? double_trace : 0.0);
          out[tensor4(a, b, c, d)] =
              packed_value(raw, 4, counts) - six / 7.0 + three / 35.0;
        }
      }
    }
  }
}

__device__ void project_rank2_spin_gradient(const double* grad, double* terms) {
  const double trace = (grad[0] + grad[4] + grad[8]) / 3.0;
  terms[0] = grad[0] - trace;
  terms[1] = grad[4] - trace;
  terms[2] = grad[8] - trace;
  terms[3] = grad[1] + grad[3];
  terms[4] = grad[2] + grad[6];
  terms[5] = grad[5] + grad[7];
}

__device__ void fill_spin_term_derivatives(
    int degree,
    int count,
    const double* terms,
    double* derivatives) {
  const int lower_count = degree == 3 ? kSpinDeg2Count : kSpinDeg3Count;
  for (int k = 0; k < 3 * lower_count; ++k) {
    derivatives[k] = 0.0;
  }
  const int exp3[kSpinDeg3Count][3] = {
      {3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {2, 1, 0}, {2, 0, 1},
      {1, 2, 0}, {0, 2, 1}, {1, 0, 2}, {0, 1, 2}, {1, 1, 1}};
  const int exp4[kSpinDeg4Count][3] = {
      {4, 0, 0}, {0, 4, 0}, {0, 0, 4}, {3, 1, 0}, {3, 0, 1},
      {1, 3, 0}, {0, 3, 1}, {1, 0, 3}, {0, 1, 3}, {2, 2, 0},
      {2, 0, 2}, {0, 2, 2}, {2, 1, 1}, {1, 2, 1}, {1, 1, 2}};
  for (int k = 0; k < count; ++k) {
    const int* exps = degree == 3 ? exp3[k] : exp4[k];
    for (int axis = 0; axis < 3; ++axis) {
      const int power = exps[axis];
      if (power == 0) {
        continue;
      }
      int lower[3] = {exps[0], exps[1], exps[2]};
      --lower[axis];
      const int lower_index =
          spin_monomial_index(degree - 1, lower[0], lower[1], lower[2]);
      derivatives[axis * lower_count + lower_index] += power * terms[k];
    }
  }
}

__device__ void add_stf_outer_gradient(
    const double* grad,
    const double* a,
    const double* b,
    double* grad_a,
    double* grad_b) {
  const double trace_grad = (grad[0] + grad[4] + grad[8]) / 3.0;
  for (int p = 0; p < 3; ++p) {
    double ga = -trace_grad * b[p];
    double gb = -trace_grad * a[p];
    for (int q = 0; q < 3; ++q) {
      ga += 0.5 * (grad[3 * p + q] + grad[3 * q + p]) * b[q];
      gb += 0.5 * (grad[3 * p + q] + grad[3 * q + p]) * a[q];
    }
    grad_a[p] += ga;
    grad_b[p] += gb;
  }
}

__device__ void add_density(
    double* density,
    int c_count,
    int width,
    const double* values,
    const double* weights,
    double scale = 1.0) {
  for (int c = 0; c < c_count; ++c) {
    const double weight = scale * weights[c];
    for (int k = 0; k < width; ++k) {
      density[c * width + k] += weight * values[k];
    }
  }
}

__device__ void load_spin_edge(
    int atom,
    int neighbor,
    int atom_stride,
    SimulationBox box,
    const double* positions_soa3,
    const double* spins_soa3,
    double* rhat,
    double& dist,
    double* si,
    double* sj) {
  float dx = 0.0f;
  float dy = 0.0f;
  float dz = 0.0f;
  minimum_image_delta(
      box,
      positions_soa3[neighbor] - positions_soa3[atom],
      positions_soa3[atom_stride + neighbor] - positions_soa3[atom_stride + atom],
      positions_soa3[2 * atom_stride + neighbor] - positions_soa3[2 * atom_stride + atom],
      dx,
      dy,
      dz);
  dist = sqrt(static_cast<double>(dx) * dx + static_cast<double>(dy) * dy +
              static_cast<double>(dz) * dz);
  if (dist > 0.0) {
    rhat[0] = dx / dist;
    rhat[1] = dy / dist;
    rhat[2] = dz / dist;
  } else {
    rhat[0] = 0.0;
    rhat[1] = 0.0;
    rhat[2] = 0.0;
  }
  si[0] = spins_soa3[atom];
  si[1] = spins_soa3[atom_stride + atom];
  si[2] = spins_soa3[2 * atom_stride + atom];
  sj[0] = spins_soa3[neighbor];
  sj[1] = spins_soa3[atom_stride + neighbor];
  sj[2] = spins_soa3[2 * atom_stride + neighbor];
}

__device__ void load_spin_edge_cached(
    int atom,
    int neighbor,
    int atom_stride,
    int slot,
    const double* spins_soa3,
    const float* spin_edge_dx,
    const float* spin_edge_dy,
    const float* spin_edge_dz,
    const float* spin_edge_dist,
    double* rhat,
    double& dist,
    double* si,
    double* sj) {
  const int offset = atom + atom_stride * slot;
  const double dx = static_cast<double>(spin_edge_dx[offset]);
  const double dy = static_cast<double>(spin_edge_dy[offset]);
  const double dz = static_cast<double>(spin_edge_dz[offset]);
  dist = static_cast<double>(spin_edge_dist[offset]);
  if (dist > 0.0) {
    rhat[0] = dx / dist;
    rhat[1] = dy / dist;
    rhat[2] = dz / dist;
  } else {
    rhat[0] = 0.0;
    rhat[1] = 0.0;
    rhat[2] = 0.0;
  }
  si[0] = spins_soa3[atom];
  si[1] = spins_soa3[atom_stride + atom];
  si[2] = spins_soa3[2 * atom_stride + atom];
  sj[0] = spins_soa3[neighbor];
  sj[1] = spins_soa3[atom_stride + neighbor];
  sj[2] = spins_soa3[2 * atom_stride + neighbor];
}

__global__ void precompute_spin_edge_weights_c4(
    int atom_count,
    int atom_stride,
    int radial_capacity,
    int num_types,
    float spin_cutoff,
    SimulationBox box,
    const int* __restrict__ types,
    const double* __restrict__ positions_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ descriptor_coefficients,
    int spin_coefficient_offset,
    float* __restrict__ spin_edge_dx,
    float* __restrict__ spin_edge_dy,
    float* __restrict__ spin_edge_dz,
    float* __restrict__ spin_edge_dist,
    float* __restrict__ spin_edge_weights,
    float* __restrict__ spin_edge_weight_derivatives) {
  constexpr int C = 4;
  constexpr int BasisCount = 4;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = atom_count * radial_capacity;
  if (index >= total) {
    return;
  }
  const int atom = index % atom_count;
  const int slot = index / atom_count;
  if (slot >= nn_radial[atom]) {
    return;
  }

  const int neighbor = nl_radial[atom + atom_stride * slot];
  float dx = 0.0f;
  float dy = 0.0f;
  float dz = 0.0f;
  minimum_image_delta(
      box,
      positions_soa3[neighbor] - positions_soa3[atom],
      positions_soa3[atom_stride + neighbor] - positions_soa3[atom_stride + atom],
      positions_soa3[2 * atom_stride + neighbor] - positions_soa3[2 * atom_stride + atom],
      dx,
      dy,
      dz);
  const float dist = sqrtf(dx * dx + dy * dy + dz * dz);
  const int edge_offset = atom + atom_stride * slot;
  spin_edge_dx[edge_offset] = dx;
  spin_edge_dy[edge_offset] = dy;
  spin_edge_dz[edge_offset] = dz;
  spin_edge_dist[edge_offset] = dist;
  if (dist <= 1.0e-12f || dist >= spin_cutoff) {
    return;
  }

  constexpr float Pi = 3.14159265358979323846f;
  const float rcinv = 1.0f / spin_cutoff;
  const float r_scaled = dist * rcinv;
  const float cutoff_phase = Pi * r_scaled;
  const float fc = 0.5f * cosf(cutoff_phase) + 0.5f;
  const float fcp = -0.5f * Pi * sinf(cutoff_phase) * rcinv;
  const float shifted = r_scaled - 1.0f;
  const float x = 2.0f * shifted * shifted - 1.0f;
  const float radial_derivative = 2.0f * shifted * rcinv;
  const float t2 = 2.0f * x * x - 1.0f;
  const float t3 = 2.0f * x * t2 - x;
  const float fn_raw[BasisCount] = {
      1.0f,
      0.5f * (x + 1.0f),
      0.5f * (t2 + 1.0f),
      0.5f * (t3 + 1.0f)};
  const float u1 = 2.0f * x;
  const float u2 = 2.0f * x * u1 - 1.0f;
  const float fn[BasisCount] = {
      fc,
      fn_raw[1] * fc,
      fn_raw[2] * fc,
      fn_raw[3] * fc};
  const float fnp[BasisCount] = {
      fcp,
      radial_derivative * fc + fn_raw[1] * fcp,
      2.0f * u1 * radial_derivative * fc + fn_raw[2] * fcp,
      3.0f * u2 * radial_derivative * fc + fn_raw[3] * fcp};

  const int type_pair = types[atom] * num_types + types[neighbor];
  for (int c = 0; c < C; ++c) {
    float weight = 0.0f;
    float derivative = 0.0f;
    #pragma unroll
    for (int k = 0; k < BasisCount; ++k) {
      const int coefficient_index = spin_coefficient_offset +
          ((c * BasisCount + k) * num_types * num_types + type_pair);
      const float coefficient = descriptor_coefficients[coefficient_index];
      weight += fn[k] * coefficient;
      derivative += fnp[k] * coefficient;
    }
    const int cache_index = spin_edge_cache_index(atom_stride, slot, atom, c);
    spin_edge_weights[cache_index] = weight;
    spin_edge_weight_derivatives[cache_index] = derivative;
  }
}

__global__ void build_spin_descriptors(
    int atom_count,
    int atom_stride,
    int struct_dim,
    int spin_dim,
    int num_types,
    int spin_compress,
    int spin_basis_size,
    int spin_l_max,
    int spin_chiral,
    float spin_cutoff,
    SimulationBox box,
    const int* __restrict__ types,
    const double* __restrict__ positions_soa3,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ descriptor_coefficients,
    int spin_coefficient_offset,
    float* __restrict__ density_rho0_cache,
    float* __restrict__ density_raw1_cache,
    float* __restrict__ density_l1_rdot_cache,
    float* __restrict__ density_l1_cross_cache,
    float* __restrict__ density_l1_stf_cache,
    float* __restrict__ density_angular2_cache,
    float* __restrict__ density_angular3_cache,
    float* __restrict__ density_angular4_cache,
    float* __restrict__ density_geom_cache,
    float* __restrict__ density_rho0_dot_cache,
    float* __restrict__ density_raw1_dot_cache,
    float* __restrict__ chiral_polar_cache,
    float* __restrict__ chiral_octupoles_raw_cache,
    float* __restrict__ chiral_hexadecapoles_raw_cache,
    float* __restrict__ chiral_chirals_cache,
    float* __restrict__ chiral_pseudodevs_cache,
    float* __restrict__ descriptors) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const double sx = spins_soa3[atom];
  const double sy = spins_soa3[atom_stride + atom];
  const double sz = spins_soa3[2 * atom_stride + atom];
  const double spin_i[3] = {sx, sy, sz};
  const double s2 = sx * sx + sy * sy + sz * sz;
  descriptors[atom + atom_stride * struct_dim] = static_cast<float>(s2);
  descriptors[atom + atom_stride * (struct_dim + 1)] =
      static_cast<float>(s2 * s2);

  double scalar_q[4 * kMaxSpinCompress] = {};
  double rho0[kMaxSpinCompress * 3] = {};
  double raw1[kMaxSpinCompress * 9] = {};
  double l1_rdot[kMaxSpinCompress] = {};
  double l1_cross[kMaxSpinCompress * 3] = {};
  double l1_stf[kMaxSpinCompress * 9] = {};
  double angular2[kMaxSpinCompress * 15] = {};
  double angular3[kMaxSpinCompress * 21] = {};
  double angular4[kMaxSpinCompress * 27] = {};
  double geom[kMaxSpinCompress * 9] = {};
  double rho0_dot[kMaxSpinCompress * 3] = {};
  double raw1_dot[kMaxSpinCompress * 9] = {};
  double polars[kMaxSpinCompress * 3] = {};
  double octupoles_raw[2 * kSpinDeg3Count] = {};
  double hexadecapoles_raw[2 * kSpinDeg4Count] = {};
  double octupoles[2 * 27] = {};
  double hexadecapoles[2 * 81] = {};
  double chirals[2] = {};
  double pseudodevs[kMaxSpinCompress * 9] = {};

  const int type1 = types[atom];
  const int type_pair_base = type1 * num_types;
  const double rcinv = 1.0 / static_cast<double>(spin_cutoff);
  const int basis_count = spin_basis_size + 1;
  const int radial_count = nn_radial[atom];
  const int chi_c = spin_compress < 2 ? spin_compress : 2;

  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    double rhat[3];
    double dist = 0.0;
    double si[3];
    double sj[3];
    load_spin_edge(atom, neighbor, atom_stride, box, positions_soa3, spins_soa3,
                   rhat, dist, si, sj);
    if (dist <= 1.0e-12 || dist >= spin_cutoff) {
      continue;
    }

    const int type_pair = type_pair_base + types[neighbor];
    double fc = 0.0;
    double fcp = 0.0;
    double fn[kMaxSpinBasis] = {};
    double fnp[kMaxSpinBasis] = {};
    find_fc_and_fcp(spin_cutoff, rcinv, dist, fc, fcp);
    find_fn_and_fnp(spin_basis_size, rcinv, dist, fc, fcp, fn, fnp);
    double weights[kMaxSpinCompress] = {};
    for (int c = 0; c < spin_compress; ++c) {
      double w = 0.0;
      for (int k = 0; k < basis_count; ++k) {
        const int index = spin_coefficient_offset +
            ((c * basis_count + k) * num_types * num_types + type_pair);
        w += fn[k] * static_cast<double>(descriptor_coefficients[index]);
      }
      weights[c] = w;
    }

    const double dot = dot3(si, sj);
    const double sj2 = dot3(sj, sj);
    const double ri_dot_si = dot3(rhat, si);
    const double ri_dot_sj = dot3(rhat, sj);
    const double bond_axis = ri_dot_si * ri_dot_sj;

    int offset = 2;
    const double scalars[4] = {dot, dot * dot, sj2, bond_axis};
    for (int term = 0; term < 4; ++term) {
      for (int c = 0; c < spin_compress; ++c) {
        scalar_q[term * kMaxSpinCompress + c] += weights[c] * scalars[term];
      }
      offset += spin_compress;
    }

    add_density(rho0, spin_compress, 3, sj, weights);
    double raw_value[9];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        raw_value[3 * a + b] = rhat[a] * sj[b];
      }
    }
    add_density(raw1, spin_compress, 9, raw_value, weights);

    if (spin_l_max >= 1) {
      const double rdot = ri_dot_sj;
      add_density(l1_rdot, spin_compress, 1, &rdot, weights);
      double cross_value[3];
      cross3(rhat, sj, cross_value);
      add_density(l1_cross, spin_compress, 3, cross_value, weights);
      double stf[9];
      stf_outer3(rhat, sj, stf);
      add_density(l1_stf, spin_compress, 9, stf, weights);
    }
    for (int ell = 2; ell <= spin_l_max; ++ell) {
      double ylm[9];
      const int ylm_width = real_spherical_harmonics_spin(rhat, ell, ylm);
      double values[27];
      int width = 0;
      for (int m = 0; m < ylm_width; ++m) {
        values[width++] = ylm[m] * sj[0];
        values[width++] = ylm[m] * sj[1];
        values[width++] = ylm[m] * sj[2];
      }
      add_density(
          ell == 2 ? angular2 : ell == 3 ? angular3 : angular4,
          spin_compress,
          width,
          values,
          weights);
    }
    double rr[9];
    stf_outer3(rhat, rhat, rr);
    add_density(geom, spin_compress, 9, rr, weights);
    if (spin_chiral) {
      add_density(polars, spin_compress, 3, rhat, weights);
      double m3[kSpinDeg3Count];
      double m4[kSpinDeg4Count];
      fill_spin_monomials(rhat, m3, m4);
      add_density(octupoles_raw, chi_c, kSpinDeg3Count, m3, weights);
      add_density(hexadecapoles_raw, chi_c, kSpinDeg4Count, m4, weights);
    }
    add_density(rho0_dot, spin_compress, 3, sj, weights, dot);
    add_density(raw1_dot, spin_compress, 9, raw_value, weights, dot);
  }

  for (int c = 0; c < spin_compress; ++c) {
    for (int k = 0; k < 3; ++k) {
      density_rho0_cache[atom + atom_stride * (c * 3 + k)] =
          rho0[idx2(c, k, 3)];
      density_rho0_dot_cache[atom + atom_stride * (c * 3 + k)] =
          rho0_dot[idx2(c, k, 3)];
    }
    for (int k = 0; k < 9; ++k) {
      density_raw1_cache[atom + atom_stride * (c * 9 + k)] =
          raw1[idx2(c, k, 9)];
      density_l1_stf_cache[atom + atom_stride * (c * 9 + k)] =
          l1_stf[idx2(c, k, 9)];
      density_geom_cache[atom + atom_stride * (c * 9 + k)] =
          geom[idx2(c, k, 9)];
      density_raw1_dot_cache[atom + atom_stride * (c * 9 + k)] =
          raw1_dot[idx2(c, k, 9)];
    }
    density_l1_rdot_cache[atom + atom_stride * c] = l1_rdot[c];
    for (int k = 0; k < 3; ++k) {
      density_l1_cross_cache[atom + atom_stride * (c * 3 + k)] =
          l1_cross[idx2(c, k, 3)];
    }
    if (density_angular2_cache != nullptr) {
      for (int k = 0; k < 15; ++k) {
        density_angular2_cache[atom + atom_stride * (c * 15 + k)] =
            angular2[idx2(c, k, 15)];
      }
    }
    if (density_angular3_cache != nullptr) {
      for (int k = 0; k < 21; ++k) {
        density_angular3_cache[atom + atom_stride * (c * 21 + k)] =
            angular3[idx2(c, k, 21)];
      }
    }
    if (density_angular4_cache != nullptr) {
      for (int k = 0; k < 27; ++k) {
        density_angular4_cache[atom + atom_stride * (c * 27 + k)] =
            angular4[idx2(c, k, 27)];
      }
    }
  }

  for (int term = 0; term < 4; ++term) {
    for (int c = 0; c < spin_compress; ++c) {
      descriptors[
          atom + atom_stride *
              (struct_dim + 2 + term * spin_compress + c)] =
          static_cast<float>(scalar_q[term * kMaxSpinCompress + c]);
    }
  }

  int offset = 2 + 4 * spin_compress;
  for (int c = 0; c < spin_compress; ++c) {
    double value = 0.0;
    for (int k = 0; k < 3; ++k) {
      value += rho0[idx2(c, k, 3)] * rho0[idx2(c, k, 3)];
    }
    descriptors[atom + atom_stride * (struct_dim + offset + c)] =
        static_cast<float>(value);
  }
  offset += spin_compress;
  if (spin_l_max >= 1) {
    for (int c = 0; c < spin_compress; ++c) {
      descriptors[atom + atom_stride * (struct_dim + offset + c)] =
          static_cast<float>(l1_rdot[c] * l1_rdot[c]);
    }
    offset += spin_compress;
    for (int c = 0; c < spin_compress; ++c) {
      double value = 0.0;
      for (int k = 0; k < 3; ++k) {
        value += l1_cross[idx2(c, k, 3)] * l1_cross[idx2(c, k, 3)];
      }
      descriptors[atom + atom_stride * (struct_dim + offset + c)] =
          static_cast<float>(value);
    }
    offset += spin_compress;
    for (int c = 0; c < spin_compress; ++c) {
      double value = 0.0;
      for (int k = 0; k < 9; ++k) {
        value += l1_stf[idx2(c, k, 9)] * l1_stf[idx2(c, k, 9)];
      }
      descriptors[atom + atom_stride * (struct_dim + offset + c)] =
          static_cast<float>(value);
    }
    offset += spin_compress;
  }
  for (int ell = 2; ell <= spin_l_max; ++ell) {
    const int width = (2 * ell + 1) * 3;
    const double* angular = ell == 2 ? angular2 : ell == 3 ? angular3 : angular4;
    for (int c = 0; c < spin_compress; ++c) {
      double value = 0.0;
      for (int k = 0; k < width; ++k) {
        value += angular[idx2(c, k, width)] * angular[idx2(c, k, width)];
      }
      descriptors[atom + atom_stride * (struct_dim + offset + c)] =
          static_cast<float>(value);
    }
    offset += spin_compress;
  }
  for (int c = 0; c < spin_compress; ++c) {
    double value = 0.0;
    const double* g = geom + c * 9;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        value += spin_i[a] * g[3 * a + b] * spin_i[b];
      }
    }
    descriptors[atom + atom_stride * (struct_dim + offset + c)] =
        static_cast<float>(value);
  }
  offset += spin_compress;
  for (int c = 0; c < spin_compress; ++c) {
    double value = 0.0;
    for (int k = 0; k < 3; ++k) {
      value += rho0[idx2(c, k, 3)] * rho0_dot[idx2(c, k, 3)];
    }
    descriptors[atom + atom_stride * (struct_dim + offset + c)] =
        static_cast<float>(value);
  }
  offset += spin_compress;
  if (spin_l_max >= 1) {
    for (int c = 0; c < spin_compress; ++c) {
      double value = 0.0;
      for (int k = 0; k < 9; ++k) {
        value += raw1[idx2(c, k, 9)] * raw1_dot[idx2(c, k, 9)];
      }
      descriptors[atom + atom_stride * (struct_dim + offset + c)] =
          static_cast<float>(value);
    }
    offset += spin_compress;
  }

  if (spin_chiral) {
    double chiral_q[2 + 2 * kMaxSpinCompress] = {};
    const int chiral_base_offset = offset;
    for (int c = 0; c < chi_c; ++c) {
      unpack_rank3_spin_stf(octupoles_raw + c * kSpinDeg3Count, octupoles + c * 27);
      unpack_rank4_spin_stf(hexadecapoles_raw + c * kSpinDeg4Count, hexadecapoles + c * 81);
      const double* qmat = geom + c * 9;
      const double* oct = octupoles + c * 27;
      const double* hex = hexadecapoles + c * 81;
      double value = 0.0;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          for (int cc = 0; cc < 3; ++cc) {
            const int eps = levi_civita(a, b, cc);
            if (eps == 0) {
              continue;
            }
            for (int d = 0; d < 3; ++d) {
              for (int e = 0; e < 3; ++e) {
                for (int f = 0; f < 3; ++f) {
                  value += eps * qmat[3 * a + d] * oct[tensor3(b, e, f)] *
                           hex[tensor4(cc, d, e, f)];
                }
              }
            }
          }
        }
      }
      chirals[c] = value;
    }

    for (int slot = 0; slot < radial_count; ++slot) {
      const int neighbor = nl_radial[atom + atom_stride * slot];
      double rhat[3];
      double dist = 0.0;
      double si[3];
      double sj[3];
      load_spin_edge(atom, neighbor, atom_stride, box, positions_soa3, spins_soa3,
                     rhat, dist, si, sj);
      if (dist <= 1.0e-12 || dist >= spin_cutoff) {
        continue;
      }
      const int type_pair = type_pair_base + types[neighbor];
      double fc = 0.0;
      double fcp = 0.0;
      double fn[kMaxSpinBasis] = {};
      double fnp[kMaxSpinBasis] = {};
      find_fc_and_fcp(spin_cutoff, rcinv, dist, fc, fcp);
      find_fn_and_fnp(spin_basis_size, rcinv, dist, fc, fcp, fn, fnp);
      double weights[kMaxSpinCompress] = {};
      for (int c = 0; c < spin_compress; ++c) {
        for (int k = 0; k < basis_count; ++k) {
          const int index = spin_coefficient_offset +
              ((c * basis_count + k) * num_types * num_types + type_pair);
          weights[c] += fn[k] * static_cast<double>(descriptor_coefficients[index]);
        }
      }
      for (int c = 0; c < spin_compress; ++c) {
        const double* qmat = geom + c * 9;
        double qu[3] = {0.0, 0.0, 0.0};
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            qu[a] += qmat[3 * a + b] * rhat[b];
          }
        }
        double axis[3];
        cross3(rhat, qu, axis);
        double pseudo[9];
        stf_outer3(axis, rhat, pseudo);
        for (int k = 0; k < 9; ++k) {
          pseudodevs[c * 9 + k] += weights[c] * pseudo[k];
        }
      }
    }

    for (int c = 0; c < spin_compress; ++c) {
      for (int k = 0; k < 3; ++k) {
        chiral_polar_cache[atom + atom_stride * (c * 3 + k)] =
            static_cast<float>(polars[idx2(c, k, 3)]);
      }
      for (int k = 0; k < 9; ++k) {
        chiral_pseudodevs_cache[atom + atom_stride * (c * 9 + k)] =
            static_cast<float>(pseudodevs[idx2(c, k, 9)]);
      }
    }
    for (int c = 0; c < chi_c; ++c) {
      for (int k = 0; k < kSpinDeg3Count; ++k) {
        chiral_octupoles_raw_cache[atom + atom_stride * (c * kSpinDeg3Count + k)] =
            static_cast<float>(octupoles_raw[c * kSpinDeg3Count + k]);
      }
      for (int k = 0; k < kSpinDeg4Count; ++k) {
        chiral_hexadecapoles_raw_cache[
            atom + atom_stride * (c * kSpinDeg4Count + k)] =
            static_cast<float>(hexadecapoles_raw[c * kSpinDeg4Count + k]);
      }
      chiral_chirals_cache[atom + atom_stride * c] =
          static_cast<float>(chirals[c]);
    }

    for (int slot = 0; slot < radial_count; ++slot) {
      const int neighbor = nl_radial[atom + atom_stride * slot];
      double rhat[3];
      double dist = 0.0;
      double si[3];
      double sj[3];
      load_spin_edge(atom, neighbor, atom_stride, box, positions_soa3, spins_soa3,
                     rhat, dist, si, sj);
      if (dist <= 1.0e-12 || dist >= spin_cutoff) {
        continue;
      }
      const int type_pair = type_pair_base + types[neighbor];
      double fc = 0.0;
      double fcp = 0.0;
      double fn[kMaxSpinBasis] = {};
      double fnp[kMaxSpinBasis] = {};
      find_fc_and_fcp(spin_cutoff, rcinv, dist, fc, fcp);
      find_fn_and_fnp(spin_basis_size, rcinv, dist, fc, fcp, fn, fnp);
      double weights[kMaxSpinCompress] = {};
      for (int c = 0; c < spin_compress; ++c) {
        for (int k = 0; k < basis_count; ++k) {
          const int index = spin_coefficient_offset +
              ((c * basis_count + k) * num_types * num_types + type_pair);
          weights[c] += fn[k] * static_cast<double>(descriptor_coefficients[index]);
        }
      }
      double spin_cross[3];
      cross3(si, sj, spin_cross);
      for (int c = 0; c < chi_c; ++c) {
        chiral_q[c] += weights[c] * dot3(spin_cross, rhat) * chirals[c];
      }
      int chiral_offset = chi_c;
      for (int c = 0; c < spin_compress; ++c) {
        const double* polar = polars + c * 3;
        double axis[3];
        cross3(polar, rhat, axis);
        chiral_q[chiral_offset + c] += weights[c] * dot3(spin_cross, axis);
      }
      chiral_offset += spin_compress;
      for (int c = 0; c < spin_compress; ++c) {
        const double* p = pseudodevs + c * 9;
        double axis[3] = {0.0, 0.0, 0.0};
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            axis[a] += p[3 * a + b] * rhat[b];
          }
        }
        chiral_q[chiral_offset + c] += weights[c] * dot3(spin_cross, axis);
      }
    }
    const int chiral_dim = chi_c + 2 * spin_compress;
    for (int d = 0; d < chiral_dim; ++d) {
      descriptors[atom + atom_stride * (struct_dim + chiral_base_offset + d)] =
          static_cast<float>(chiral_q[d]);
    }
  }
}

__global__ void build_spin_descriptors_c4_l4_basic(
    int atom_count,
    int atom_stride,
    int struct_dim,
    float spin_cutoff,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ spin_edge_dx,
    const float* __restrict__ spin_edge_dy,
    const float* __restrict__ spin_edge_dz,
    const float* __restrict__ spin_edge_dist,
    const float* __restrict__ spin_edge_weights,
    float* __restrict__ density_rho0_cache,
    float* __restrict__ density_raw1_cache,
    float* __restrict__ density_l1_rdot_cache,
    float* __restrict__ density_l1_cross_cache,
    float* __restrict__ density_l1_stf_cache,
    float* __restrict__ density_angular2_cache,
    float* __restrict__ density_angular3_cache,
    float* __restrict__ density_angular4_cache,
    float* __restrict__ density_geom_cache,
    float* __restrict__ density_rho0_dot_cache,
    float* __restrict__ density_raw1_dot_cache,
    float* __restrict__ chiral_polar_cache,
    float* __restrict__ chiral_octupoles_raw_cache,
    float* __restrict__ chiral_hexadecapoles_raw_cache,
    float* __restrict__ descriptors) {
  constexpr int C = 4;
  constexpr int ChiC = 2;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int atom = tid / C;
  const int c = tid - atom * C;
  if (atom >= atom_count) {
    return;
  }

  const double sx = spins_soa3[atom];
  const double sy = spins_soa3[atom_stride + atom];
  const double sz = spins_soa3[2 * atom_stride + atom];
  const double spin_i[3] = {sx, sy, sz};
  if (c == 0) {
    const double s2 = sx * sx + sy * sy + sz * sz;
    descriptors[atom + atom_stride * struct_dim] = static_cast<float>(s2);
    descriptors[atom + atom_stride * (struct_dim + 1)] =
        static_cast<float>(s2 * s2);
  }

  double scalar_q[4] = {};
  double rho0[3] = {};
  double raw1[9] = {};
  double l1_rdot = 0.0;
  double l1_cross[3] = {};
  double l1_stf[9] = {};
  double angular2[15] = {};
  double angular3[21] = {};
  double angular4[27] = {};
  double geom[9] = {};
  double rho0_dot[3] = {};
  double raw1_dot[9] = {};
  double polar[3] = {};
  double octupoles_raw[kSpinDeg3Count] = {};
  double hexadecapoles_raw[kSpinDeg4Count] = {};

  const int radial_count = nn_radial[atom];
  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    double rhat[3];
    double dist = 0.0;
    double si[3];
    double sj[3];
    load_spin_edge_cached(
        atom,
        neighbor,
        atom_stride,
        slot,
        spins_soa3,
        spin_edge_dx,
        spin_edge_dy,
        spin_edge_dz,
        spin_edge_dist,
        rhat,
        dist,
        si,
        sj);
    if (dist <= 1.0e-12 || dist >= spin_cutoff) {
      continue;
    }

    const double weight = static_cast<double>(
        spin_edge_weights[spin_edge_cache_index(atom_stride, slot, atom, c)]);
    const double dot = dot3(si, sj);
    const double sj2 = dot3(sj, sj);
    const double ri_dot_si = dot3(rhat, si);
    const double ri_dot_sj = dot3(rhat, sj);
    const double scalars[4] = {
        dot, dot * dot, sj2, ri_dot_si * ri_dot_sj};
    for (int term = 0; term < 4; ++term) {
      scalar_q[term] += weight * scalars[term];
    }
    for (int k = 0; k < 3; ++k) {
      rho0[k] += weight * sj[k];
      rho0_dot[k] += weight * dot * sj[k];
      polar[k] += weight * rhat[k];
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        const double value = rhat[a] * sj[b];
        raw1[3 * a + b] += weight * value;
        raw1_dot[3 * a + b] += weight * dot * value;
      }
    }
    l1_rdot += weight * ri_dot_sj;
    double cross_value[3];
    cross3(rhat, sj, cross_value);
    for (int k = 0; k < 3; ++k) {
      l1_cross[k] += weight * cross_value[k];
    }
    double stf[9];
    stf_outer3(rhat, sj, stf);
    for (int k = 0; k < 9; ++k) {
      l1_stf[k] += weight * stf[k];
    }
    for (int ell = 2; ell <= 4; ++ell) {
      double ylm[9];
      const int ylm_width = real_spherical_harmonics_spin(rhat, ell, ylm);
      double* angular = ell == 2 ? angular2 : ell == 3 ? angular3 : angular4;
      int width = 0;
      for (int m = 0; m < ylm_width; ++m) {
        angular[width++] += weight * ylm[m] * sj[0];
        angular[width++] += weight * ylm[m] * sj[1];
        angular[width++] += weight * ylm[m] * sj[2];
      }
    }
    double rr[9];
    stf_outer3(rhat, rhat, rr);
    for (int k = 0; k < 9; ++k) {
      geom[k] += weight * rr[k];
    }
    if (c < ChiC) {
      double m3[kSpinDeg3Count];
      double m4[kSpinDeg4Count];
      fill_spin_monomials(rhat, m3, m4);
      for (int k = 0; k < kSpinDeg3Count; ++k) {
        octupoles_raw[k] += weight * m3[k];
      }
      for (int k = 0; k < kSpinDeg4Count; ++k) {
        hexadecapoles_raw[k] += weight * m4[k];
      }
    }
  }

  for (int k = 0; k < 3; ++k) {
    density_rho0_cache[atom + atom_stride * (c * 3 + k)] =
        static_cast<float>(rho0[k]);
    density_rho0_dot_cache[atom + atom_stride * (c * 3 + k)] =
        static_cast<float>(rho0_dot[k]);
    density_l1_cross_cache[atom + atom_stride * (c * 3 + k)] =
        static_cast<float>(l1_cross[k]);
    chiral_polar_cache[atom + atom_stride * (c * 3 + k)] =
        static_cast<float>(polar[k]);
  }
  for (int k = 0; k < 9; ++k) {
    density_raw1_cache[atom + atom_stride * (c * 9 + k)] =
        static_cast<float>(raw1[k]);
    density_l1_stf_cache[atom + atom_stride * (c * 9 + k)] =
        static_cast<float>(l1_stf[k]);
    density_geom_cache[atom + atom_stride * (c * 9 + k)] =
        static_cast<float>(geom[k]);
    density_raw1_dot_cache[atom + atom_stride * (c * 9 + k)] =
        static_cast<float>(raw1_dot[k]);
  }
  density_l1_rdot_cache[atom + atom_stride * c] =
      static_cast<float>(l1_rdot);
  for (int k = 0; k < 15; ++k) {
    density_angular2_cache[atom + atom_stride * (c * 15 + k)] =
        static_cast<float>(angular2[k]);
  }
  for (int k = 0; k < 21; ++k) {
    density_angular3_cache[atom + atom_stride * (c * 21 + k)] =
        static_cast<float>(angular3[k]);
  }
  for (int k = 0; k < 27; ++k) {
    density_angular4_cache[atom + atom_stride * (c * 27 + k)] =
        static_cast<float>(angular4[k]);
  }
  if (c < ChiC) {
    for (int k = 0; k < kSpinDeg3Count; ++k) {
      chiral_octupoles_raw_cache[
          atom + atom_stride * (c * kSpinDeg3Count + k)] =
          static_cast<float>(octupoles_raw[k]);
    }
    for (int k = 0; k < kSpinDeg4Count; ++k) {
      chiral_hexadecapoles_raw_cache[
          atom + atom_stride * (c * kSpinDeg4Count + k)] =
          static_cast<float>(hexadecapoles_raw[k]);
    }
  }

  for (int term = 0; term < 4; ++term) {
    descriptors[atom + atom_stride * (struct_dim + 2 + term * C + c)] =
        static_cast<float>(scalar_q[term]);
  }
  int offset = 2 + 4 * C;
  double value = 0.0;
  for (int k = 0; k < 3; ++k) {
    value += rho0[k] * rho0[k];
  }
  descriptors[atom + atom_stride * (struct_dim + offset + c)] =
      static_cast<float>(value);
  offset += C;
  descriptors[atom + atom_stride * (struct_dim + offset + c)] =
      static_cast<float>(l1_rdot * l1_rdot);
  offset += C;
  value = 0.0;
  for (int k = 0; k < 3; ++k) {
    value += l1_cross[k] * l1_cross[k];
  }
  descriptors[atom + atom_stride * (struct_dim + offset + c)] =
      static_cast<float>(value);
  offset += C;
  value = 0.0;
  for (int k = 0; k < 9; ++k) {
    value += l1_stf[k] * l1_stf[k];
  }
  descriptors[atom + atom_stride * (struct_dim + offset + c)] =
      static_cast<float>(value);
  offset += C;
  const double* angular[3] = {angular2, angular3, angular4};
  const int widths[3] = {15, 21, 27};
  for (int ell_index = 0; ell_index < 3; ++ell_index) {
    value = 0.0;
    for (int k = 0; k < widths[ell_index]; ++k) {
      value += angular[ell_index][k] * angular[ell_index][k];
    }
    descriptors[atom + atom_stride * (struct_dim + offset + c)] =
        static_cast<float>(value);
    offset += C;
  }
  value = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      value += spin_i[a] * geom[3 * a + b] * spin_i[b];
    }
  }
  descriptors[atom + atom_stride * (struct_dim + offset + c)] =
      static_cast<float>(value);
  offset += C;
  value = 0.0;
  for (int k = 0; k < 3; ++k) {
    value += rho0[k] * rho0_dot[k];
  }
  descriptors[atom + atom_stride * (struct_dim + offset + c)] =
      static_cast<float>(value);
  offset += C;
  value = 0.0;
  for (int k = 0; k < 9; ++k) {
    value += raw1[k] * raw1_dot[k];
  }
  descriptors[atom + atom_stride * (struct_dim + offset + c)] =
      static_cast<float>(value);
}

__device__ __forceinline__ void load_spin_edge_cached_f32(
    int atom,
    int neighbor,
    int atom_stride,
    int slot,
    const double* spins_soa3,
    const float* spin_edge_dx,
    const float* spin_edge_dy,
    const float* spin_edge_dz,
    const float* spin_edge_dist,
    float* rhat,
    float& dist,
    float* si,
    float* sj);
__device__ __forceinline__ void cross3f(
    const float* a,
    const float* b,
    float* out);
__device__ __forceinline__ float dot3f(const float* a, const float* b);
__device__ __forceinline__ void stf_outer3f(
    const float* a,
    const float* b,
    float* out);
__device__ __forceinline__ void fill_spin_monomialsf(
    const float* u,
    float* m3,
    float* m4);
__device__ int real_spherical_harmonics_spinf(
    const float* rhat,
    int ell,
    float* out);

template <int SlotCapacity, bool AtomMajor>
__global__ void __launch_bounds__(128, 1) build_spin_primitive_cache_c4_l4_warp(
    int atom_count,
    int atom_stride,
    int struct_dim,
    float spin_cutoff,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ spin_edge_dx,
    const float* __restrict__ spin_edge_dy,
    const float* __restrict__ spin_edge_dz,
    const float* __restrict__ spin_edge_dist,
    const float* __restrict__ spin_edge_weights,
    float* __restrict__ density_rho0_cache,
    float* __restrict__ density_raw1_cache,
    float* __restrict__ density_l1_rdot_cache,
    float* __restrict__ density_l1_cross_cache,
    float* __restrict__ density_l1_stf_cache,
    float* __restrict__ density_angular2_cache,
    float* __restrict__ density_angular3_cache,
    float* __restrict__ density_angular4_cache,
    float* __restrict__ density_geom_cache,
    float* __restrict__ density_rho0_dot_cache,
    float* __restrict__ density_raw1_dot_cache,
    float* __restrict__ chiral_polar_cache,
    float* __restrict__ chiral_octupoles_raw_cache,
    float* __restrict__ chiral_hexadecapoles_raw_cache,
    float* __restrict__ descriptors) {
  constexpr int C = 4;
  constexpr int ChiC = 2;
  constexpr int ScalarComponents = 16;
  constexpr int Rho0Base = ScalarComponents;
  constexpr int Raw1Base = Rho0Base + C * 3;
  constexpr int L1RdotBase = Raw1Base + C * 9;
  constexpr int L1CrossBase = L1RdotBase + C;
  constexpr int L1StfBase = L1CrossBase + C * 3;
  constexpr int Angular2Base = L1StfBase + C * 9;
  constexpr int Angular3Base = Angular2Base + C * 15;
  constexpr int Angular4Base = Angular3Base + C * 21;
  constexpr int GeomBase = Angular4Base + C * 27;
  constexpr int Rho0DotBase = GeomBase + C * 9;
  constexpr int Raw1DotBase = Rho0DotBase + C * 3;
  constexpr int PolarBase = Raw1DotBase + C * 9;
  constexpr int OctBase = PolarBase + C * 3;
  constexpr int HexBase = OctBase + ChiC * kSpinDeg3Count;
  constexpr int ComponentCount = HexBase + ChiC * kSpinDeg4Count;
  constexpr int DensityComponentCount = PolarBase - Rho0Base;
  __shared__ float prim[kSpinPrimitiveCount][SlotCapacity + 1];
  __shared__ float weights[C][SlotCapacity + 1];
  __shared__ float density_components[DensityComponentCount];

  const int lane = threadIdx.x;
  const int atom = blockIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const float si[3] = {
      static_cast<float>(spins_soa3[atom]),
      static_cast<float>(spins_soa3[atom_stride + atom]),
      static_cast<float>(spins_soa3[2 * atom_stride + atom])};
  if (lane == 0) {
    const float s2 = dot3f(si, si);
    descriptors[atom + atom_stride * struct_dim] = s2;
    descriptors[atom + atom_stride * (struct_dim + 1)] =
        s2 * s2;
  }

  const int radial_count = nn_radial[atom];
  const int count = radial_count < SlotCapacity ? radial_count : SlotCapacity;
  for (int slot = lane; slot < count; slot += blockDim.x) {
    float rhat[3];
    float dist = 0.0f;
    float si_edge[3];
    float sj[3];
    load_spin_edge_cached_f32(
        atom,
        nl_radial[atom + atom_stride * slot],
        atom_stride,
        slot,
        spins_soa3,
        spin_edge_dx,
        spin_edge_dy,
        spin_edge_dz,
        spin_edge_dist,
        rhat,
        dist,
        si_edge,
        sj);

    for (int p = 0; p < kSpinPrimitiveCount; ++p) {
      prim[p][slot] = 0.0f;
    }
    for (int c = 0; c < C; ++c) {
      weights[c][slot] = 0.0f;
    }
    if (dist > 1.0e-12f && dist < spin_cutoff) {
      for (int c = 0; c < C; ++c) {
        weights[c][slot] = spin_edge_weights[
            spin_edge_cache_index(atom_stride, slot, atom, c)];
      }
      const float dot = dot3f(si, sj);
      const float sj2 = dot3f(sj, sj);
      const float ri_dot_si = dot3f(rhat, si);
      const float ri_dot_sj = dot3f(rhat, sj);
      prim[0][slot] = sj[0];
      prim[1][slot] = sj[1];
      prim[2][slot] = sj[2];
      prim[3][slot] = dot;
      prim[4][slot] = sj2;
      prim[5][slot] = ri_dot_si;
      prim[6][slot] = ri_dot_sj;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          prim[7 + 3 * a + b][slot] = rhat[a] * sj[b];
        }
      }
      float cross_value[3];
      cross3f(rhat, sj, cross_value);
      for (int k = 0; k < 3; ++k) {
        prim[16 + k][slot] = cross_value[k];
      }
      float stf[9];
      stf_outer3f(rhat, sj, stf);
      for (int k = 0; k < 9; ++k) {
        prim[19 + k][slot] = stf[k];
      }
      float ylm[9];
      int width = real_spherical_harmonics_spinf(rhat, 2, ylm);
      for (int m = 0; m < width; ++m) {
        for (int d = 0; d < 3; ++d) {
          prim[28 + m * 3 + d][slot] = ylm[m] * sj[d];
        }
      }
      width = real_spherical_harmonics_spinf(rhat, 3, ylm);
      for (int m = 0; m < width; ++m) {
        for (int d = 0; d < 3; ++d) {
          prim[43 + m * 3 + d][slot] = ylm[m] * sj[d];
        }
      }
      width = real_spherical_harmonics_spinf(rhat, 4, ylm);
      for (int m = 0; m < width; ++m) {
        for (int d = 0; d < 3; ++d) {
          prim[64 + m * 3 + d][slot] = ylm[m] * sj[d];
        }
      }
      float rr[9];
      stf_outer3f(rhat, rhat, rr);
      for (int k = 0; k < 9; ++k) {
        prim[91 + k][slot] = rr[k];
      }
      prim[100][slot] = rhat[0];
      prim[101][slot] = rhat[1];
      prim[102][slot] = rhat[2];
      float m3[kSpinDeg3Count];
      float m4[kSpinDeg4Count];
      fill_spin_monomialsf(rhat, m3, m4);
      for (int k = 0; k < kSpinDeg3Count; ++k) {
        prim[103 + k][slot] = m3[k];
      }
      for (int k = 0; k < kSpinDeg4Count; ++k) {
        prim[113 + k][slot] = m4[k];
      }
    }
  }
  __syncthreads();

  for (int component = lane; component < ComponentCount; component += blockDim.x) {
    int c = 0;
    int k = 0;
    int p = 0;
    float acc = 0.0f;
    if (component < ScalarComponents) {
      const int term = component / C;
      c = component - term * C;
      for (int slot = 0; slot < count; ++slot) {
        const float dot = prim[3][slot];
        const float value =
            term == 0 ? dot :
            term == 1 ? dot * dot :
            term == 2 ? prim[4][slot] : prim[5][slot] * prim[6][slot];
        acc += weights[c][slot] * value;
      }
      descriptors[atom + atom_stride * (struct_dim + 2 + term * C + c)] = acc;
      continue;
    } else if (component < Raw1Base) {
      k = component - Rho0Base;
      c = k / 3;
      p = k - c * 3;
      for (int slot = 0; slot < count; ++slot) acc += weights[c][slot] * prim[p][slot];
      density_rho0_cache[
          spin_component_cache_index<AtomMajor, C * 3>(atom_stride, atom, k)] = acc;
      density_components[component - Rho0Base] = acc;
      continue;
    } else if (component < L1RdotBase) {
      k = component - Raw1Base;
      c = k / 9;
      p = 7 + (k - c * 9);
      for (int slot = 0; slot < count; ++slot) acc += weights[c][slot] * prim[p][slot];
      density_raw1_cache[
          spin_component_cache_index<AtomMajor, C * 9>(atom_stride, atom, k)] = acc;
      density_components[component - Rho0Base] = acc;
      continue;
    } else if (component < L1CrossBase) {
      k = component - L1RdotBase;
      c = k;
      for (int slot = 0; slot < count; ++slot) acc += weights[c][slot] * prim[6][slot];
      density_l1_rdot_cache[
          spin_component_cache_index<AtomMajor, C>(atom_stride, atom, c)] = acc;
      density_components[component - Rho0Base] = acc;
      continue;
    } else if (component < L1StfBase) {
      k = component - L1CrossBase;
      c = k / 3;
      p = 16 + (k - c * 3);
      for (int slot = 0; slot < count; ++slot) acc += weights[c][slot] * prim[p][slot];
      density_l1_cross_cache[
          spin_component_cache_index<AtomMajor, C * 3>(atom_stride, atom, k)] = acc;
      density_components[component - Rho0Base] = acc;
      continue;
    } else if (component < Angular2Base) {
      k = component - L1StfBase;
      c = k / 9;
      p = 19 + (k - c * 9);
      for (int slot = 0; slot < count; ++slot) acc += weights[c][slot] * prim[p][slot];
      density_l1_stf_cache[
          spin_component_cache_index<AtomMajor, C * 9>(atom_stride, atom, k)] = acc;
      density_components[component - Rho0Base] = acc;
      continue;
    } else if (component < Angular3Base) {
      k = component - Angular2Base;
      c = k / 15;
      p = 28 + (k - c * 15);
      for (int slot = 0; slot < count; ++slot) acc += weights[c][slot] * prim[p][slot];
      density_angular2_cache[
          spin_component_cache_index<AtomMajor, C * 15>(atom_stride, atom, k)] = acc;
      density_components[component - Rho0Base] = acc;
      continue;
    } else if (component < Angular4Base) {
      k = component - Angular3Base;
      c = k / 21;
      p = 43 + (k - c * 21);
      for (int slot = 0; slot < count; ++slot) acc += weights[c][slot] * prim[p][slot];
      density_angular3_cache[
          spin_component_cache_index<AtomMajor, C * 21>(atom_stride, atom, k)] = acc;
      density_components[component - Rho0Base] = acc;
      continue;
    } else if (component < GeomBase) {
      k = component - Angular4Base;
      c = k / 27;
      p = 64 + (k - c * 27);
      for (int slot = 0; slot < count; ++slot) acc += weights[c][slot] * prim[p][slot];
      density_angular4_cache[
          spin_component_cache_index<AtomMajor, C * 27>(atom_stride, atom, k)] = acc;
      density_components[component - Rho0Base] = acc;
      continue;
    } else if (component < Rho0DotBase) {
      k = component - GeomBase;
      c = k / 9;
      p = 91 + (k - c * 9);
      for (int slot = 0; slot < count; ++slot) acc += weights[c][slot] * prim[p][slot];
      density_geom_cache[
          spin_component_cache_index<AtomMajor, C * 9>(atom_stride, atom, k)] = acc;
      density_components[component - Rho0Base] = acc;
      continue;
    } else if (component < Raw1DotBase) {
      k = component - Rho0DotBase;
      c = k / 3;
      p = k - c * 3;
      for (int slot = 0; slot < count; ++slot) {
        acc += weights[c][slot] * prim[3][slot] * prim[p][slot];
      }
      density_rho0_dot_cache[
          spin_component_cache_index<AtomMajor, C * 3>(atom_stride, atom, k)] = acc;
      density_components[component - Rho0Base] = acc;
      continue;
    } else if (component < PolarBase) {
      k = component - Raw1DotBase;
      c = k / 9;
      p = 7 + (k - c * 9);
      for (int slot = 0; slot < count; ++slot) {
        acc += weights[c][slot] * prim[3][slot] * prim[p][slot];
      }
      density_raw1_dot_cache[
          spin_component_cache_index<AtomMajor, C * 9>(atom_stride, atom, k)] = acc;
      density_components[component - Rho0Base] = acc;
      continue;
    } else if (component < OctBase) {
      k = component - PolarBase;
      c = k / 3;
      p = 100 + (k - c * 3);
      for (int slot = 0; slot < count; ++slot) acc += weights[c][slot] * prim[p][slot];
      chiral_polar_cache[
          spin_component_cache_index<AtomMajor, C * 3>(atom_stride, atom, k)] = acc;
      continue;
    } else if (component < HexBase) {
      k = component - OctBase;
      c = k / kSpinDeg3Count;
      p = 103 + (k - c * kSpinDeg3Count);
      for (int slot = 0; slot < count; ++slot) acc += weights[c][slot] * prim[p][slot];
      chiral_octupoles_raw_cache[
          spin_component_cache_index<AtomMajor, ChiC * kSpinDeg3Count>(
              atom_stride, atom, k)] = acc;
      continue;
    } else {
      k = component - HexBase;
      c = k / kSpinDeg4Count;
      p = 113 + (k - c * kSpinDeg4Count);
      for (int slot = 0; slot < count; ++slot) acc += weights[c][slot] * prim[p][slot];
      chiral_hexadecapoles_raw_cache[
          spin_component_cache_index<AtomMajor, ChiC * kSpinDeg4Count>(
              atom_stride, atom, k)] = acc;
    }
  }

  __syncthreads();
  if (lane < C) {
    const int channel = lane;
    int offset = 2 + 4 * C;
    double value = 0.0;
    for (int k = 0; k < 3; ++k) {
      const double rho0 = density_components[
          Rho0Base - Rho0Base + channel * 3 + k];
      value += rho0 * rho0;
    }
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        static_cast<float>(value);
    offset += C;

    const double rdot = density_components[
        L1RdotBase - Rho0Base + channel];
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        static_cast<float>(rdot * rdot);
    offset += C;

    value = 0.0;
    for (int k = 0; k < 3; ++k) {
      const double x = density_components[
          L1CrossBase - Rho0Base + channel * 3 + k];
      value += x * x;
    }
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        static_cast<float>(value);
    offset += C;

    value = 0.0;
    for (int k = 0; k < 9; ++k) {
      const double x = density_components[
          L1StfBase - Rho0Base + channel * 9 + k];
      value += x * x;
    }
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        static_cast<float>(value);
    offset += C;

    constexpr int AngularBases[3] = {
        Angular2Base,
        Angular3Base,
        Angular4Base};
    constexpr int AngularWidths[3] = {15, 21, 27};
    for (int ell_index = 0; ell_index < 3; ++ell_index) {
      value = 0.0;
      for (int k = 0; k < AngularWidths[ell_index]; ++k) {
        const double x = density_components[
            AngularBases[ell_index] - Rho0Base +
            channel * AngularWidths[ell_index] + k];
        value += x * x;
      }
      descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
          static_cast<float>(value);
      offset += C;
    }

    const double spin_i[3] = {
        spins_soa3[atom],
        spins_soa3[atom_stride + atom],
        spins_soa3[2 * atom_stride + atom]};
    value = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        value += spin_i[a] * density_components[
            GeomBase - Rho0Base + channel * 9 + 3 * a + b] * spin_i[b];
      }
    }
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        static_cast<float>(value);
    offset += C;

    value = 0.0;
    for (int k = 0; k < 3; ++k) {
      value += density_components[
                   Rho0Base - Rho0Base + channel * 3 + k] *
               density_components[
                   Rho0DotBase - Rho0Base + channel * 3 + k];
    }
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        static_cast<float>(value);
    offset += C;

    value = 0.0;
    for (int k = 0; k < 9; ++k) {
      value += density_components[
                   Raw1Base - Rho0Base + channel * 9 + k] *
               density_components[
                   Raw1DotBase - Rho0Base + channel * 9 + k];
    }
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        static_cast<float>(value);
  }
}

__global__ void accumulate_spin_onsite_mforces(
    int atom_count,
    int atom_stride,
    int struct_dim,
    const double* __restrict__ spins_soa3,
    const float* __restrict__ fp,
    double* __restrict__ mforce_soa3) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  const double sx = spins_soa3[atom];
  const double sy = spins_soa3[atom_stride + atom];
  const double sz = spins_soa3[2 * atom_stride + atom];
  const double s2 = sx * sx + sy * sy + sz * sz;
  const double scale =
      2.0 * static_cast<double>(fp[atom + atom_stride * struct_dim]) +
      4.0 * static_cast<double>(fp[atom + atom_stride * (struct_dim + 1)]) * s2;
  mforce_soa3[atom] -= scale * sx;
  mforce_soa3[atom_stride + atom] -= scale * sy;
  mforce_soa3[2 * atom_stride + atom] -= scale * sz;
}

__global__ void accumulate_spin_scalar_forces(
    int atom_count,
    int atom_stride,
    int struct_dim,
    int num_types,
    int spin_compress,
    int spin_basis_size,
    float spin_cutoff,
    SimulationBox box,
    const int* __restrict__ types,
    const double* __restrict__ positions_soa3,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ fp,
    const float* __restrict__ descriptor_coefficients,
    int spin_coefficient_offset,
    const float* __restrict__ spin_edge_dx,
    const float* __restrict__ spin_edge_dy,
    const float* __restrict__ spin_edge_dz,
    const float* __restrict__ spin_edge_dist,
    const float* __restrict__ spin_edge_weights,
    const float* __restrict__ spin_edge_weight_derivatives,
    bool use_cached_geometry,
    double* __restrict__ force_soa3,
    double* __restrict__ mforce_soa3,
    bool accumulate_virial,
    double* __restrict__ virial_soa9) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const int type1 = types[atom];
  const int type_pair_base = type1 * num_types;
  const int basis_count = spin_basis_size + 1;
  const double rcinv = 1.0 / static_cast<double>(spin_cutoff);
  const int radial_count = nn_radial[atom];

  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    double rhat[3];
    double dist = 0.0;
    double si[3];
    double sj[3];
    if (use_cached_geometry) {
      load_spin_edge_cached(
          atom,
          neighbor,
          atom_stride,
          slot,
          spins_soa3,
          spin_edge_dx,
          spin_edge_dy,
          spin_edge_dz,
          spin_edge_dist,
          rhat,
          dist,
          si,
          sj);
    } else {
      load_spin_edge(atom, neighbor, atom_stride, box, positions_soa3, spins_soa3,
                     rhat, dist, si, sj);
    }
    if (dist <= 1.0e-12 || dist >= spin_cutoff) {
      continue;
    }

    double weights[kMaxSpinCompress] = {};
    double weight_derivatives[kMaxSpinCompress] = {};
    if (spin_edge_weights != nullptr && spin_edge_weight_derivatives != nullptr &&
        spin_compress == 4 && spin_basis_size == 3) {
      for (int c = 0; c < 4; ++c) {
        const int cache_index = spin_edge_cache_index(atom_stride, slot, atom, c);
        weights[c] = static_cast<double>(spin_edge_weights[cache_index]);
        weight_derivatives[c] =
            static_cast<double>(spin_edge_weight_derivatives[cache_index]);
      }
    } else {
      const int type_pair = type_pair_base + types[neighbor];
      double fc = 0.0;
      double fcp = 0.0;
      double fn[kMaxSpinBasis] = {};
      double fnp[kMaxSpinBasis] = {};
      find_fc_and_fcp(spin_cutoff, rcinv, dist, fc, fcp);
      find_fn_and_fnp(spin_basis_size, rcinv, dist, fc, fcp, fn, fnp);
      for (int c = 0; c < spin_compress; ++c) {
        for (int k = 0; k < basis_count; ++k) {
          const int index = spin_coefficient_offset +
              ((c * basis_count + k) * num_types * num_types + type_pair);
          const double coefficient =
              static_cast<double>(descriptor_coefficients[index]);
          weights[c] += fn[k] * coefficient;
          weight_derivatives[c] += fnp[k] * coefficient;
        }
      }
    }

    const double dot = dot3(si, sj);
    const double sj2 = dot3(sj, sj);
    const double ri_dot_si = dot3(rhat, si);
    const double ri_dot_sj = dot3(rhat, sj);
    const double bond_axis = ri_dot_si * ri_dot_sj;
    const double scalars[4] = {dot, dot * dot, sj2, bond_axis};

    double grad_weight[kMaxSpinCompress] = {};
    double grad_rhat[3] = {0.0, 0.0, 0.0};
    double grad_si[3] = {0.0, 0.0, 0.0};
    double grad_sj[3] = {0.0, 0.0, 0.0};
    double grad_dot = 0.0;

    int offset = 2;
    double g = 0.0;
    for (int c = 0; c < spin_compress; ++c) {
      const double alpha =
          static_cast<double>(fp[atom + atom_stride * (struct_dim + offset + c)]);
      grad_weight[c] += alpha * scalars[0];
      g += alpha * weights[c];
    }
    grad_dot += g;
    offset += spin_compress;

    g = 0.0;
    for (int c = 0; c < spin_compress; ++c) {
      const double alpha =
          static_cast<double>(fp[atom + atom_stride * (struct_dim + offset + c)]);
      grad_weight[c] += alpha * scalars[1];
      g += alpha * weights[c];
    }
    grad_dot += 2.0 * dot * g;
    offset += spin_compress;

    g = 0.0;
    for (int c = 0; c < spin_compress; ++c) {
      const double alpha =
          static_cast<double>(fp[atom + atom_stride * (struct_dim + offset + c)]);
      grad_weight[c] += alpha * scalars[2];
      g += alpha * weights[c];
    }
    for (int d = 0; d < 3; ++d) {
      grad_sj[d] += 2.0 * g * sj[d];
    }
    offset += spin_compress;

    g = 0.0;
    for (int c = 0; c < spin_compress; ++c) {
      const double alpha =
          static_cast<double>(fp[atom + atom_stride * (struct_dim + offset + c)]);
      grad_weight[c] += alpha * scalars[3];
      g += alpha * weights[c];
    }
    for (int d = 0; d < 3; ++d) {
      grad_si[d] += g * ri_dot_sj * rhat[d];
      grad_sj[d] += g * ri_dot_si * rhat[d];
      grad_rhat[d] += g * (ri_dot_sj * si[d] + ri_dot_si * sj[d]);
    }

    for (int d = 0; d < 3; ++d) {
      grad_si[d] += grad_dot * sj[d];
      grad_sj[d] += grad_dot * si[d];
    }

    double grad_dist = 0.0;
    for (int c = 0; c < spin_compress; ++c) {
      grad_dist += grad_weight[c] * weight_derivatives[c];
    }
    double dot_r = 0.0;
    for (int d = 0; d < 3; ++d) {
      dot_r += grad_rhat[d] * rhat[d];
    }
    double grad_rij[3];
    for (int d = 0; d < 3; ++d) {
      grad_rij[d] = grad_dist * rhat[d] +
                    (grad_rhat[d] - dot_r * rhat[d]) / dist;
      atomicAdd(force_soa3 + d * atom_stride + atom, grad_rij[d]);
      atomicAdd(force_soa3 + d * atom_stride + neighbor, -grad_rij[d]);
      atomicAdd(mforce_soa3 + d * atom_stride + atom, -grad_si[d]);
      atomicAdd(mforce_soa3 + d * atom_stride + neighbor, -grad_sj[d]);
    }
    if (accumulate_virial) {
      for (int a = 0; a < 3; ++a) {
        const double rij_a = rhat[a] * dist;
        for (int b = 0; b < 3; ++b) {
          const int row_major = a * 3 + b;
          const int internal_component =
              row_major == 0 ? 0 :
              row_major == 1 ? 3 :
              row_major == 2 ? 4 :
              row_major == 3 ? 6 :
              row_major == 4 ? 1 :
              row_major == 5 ? 5 :
              row_major == 6 ? 7 :
              row_major == 7 ? 8 : 2;
          atomicAdd(
              virial_soa9 + internal_component * atom_stride + atom,
              -rij_a * grad_rij[b]);
        }
      }
    }
  }
}

__global__ void accumulate_spin_density_forces(
    int atom_count,
    int atom_stride,
    int struct_dim,
    int num_types,
    int spin_compress,
    int spin_basis_size,
    int spin_l_max,
    float spin_cutoff,
    SimulationBox box,
    const int* __restrict__ types,
    const double* __restrict__ positions_soa3,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ fp,
    const float* __restrict__ descriptor_coefficients,
    int spin_coefficient_offset,
    const float* __restrict__ spin_edge_dx,
    const float* __restrict__ spin_edge_dy,
    const float* __restrict__ spin_edge_dz,
    const float* __restrict__ spin_edge_dist,
    const float* __restrict__ spin_edge_weights,
    const float* __restrict__ spin_edge_weight_derivatives,
    bool use_cached_geometry,
    const float* __restrict__ density_rho0_cache,
    const float* __restrict__ density_raw1_cache,
    const float* __restrict__ density_l1_rdot_cache,
    const float* __restrict__ density_l1_cross_cache,
    const float* __restrict__ density_l1_stf_cache,
    const float* __restrict__ density_angular2_cache,
    const float* __restrict__ density_angular3_cache,
    const float* __restrict__ density_angular4_cache,
    const float* __restrict__ density_geom_cache,
    const float* __restrict__ density_rho0_dot_cache,
    const float* __restrict__ density_raw1_dot_cache,
    double* __restrict__ force_soa3,
    double* __restrict__ mforce_soa3,
    bool accumulate_virial,
    double* __restrict__ virial_soa9) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  double rho0[kMaxSpinCompress * 3] = {};
  double raw1[kMaxSpinCompress * 9] = {};
  double l1_rdot[kMaxSpinCompress] = {};
  double l1_cross[kMaxSpinCompress * 3] = {};
  double l1_stf[kMaxSpinCompress * 9] = {};
  double angular2[kMaxSpinCompress * 15] = {};
  double angular3[kMaxSpinCompress * 21] = {};
  double angular4[kMaxSpinCompress * 27] = {};
  double geom[kMaxSpinCompress * 9] = {};
  double rho0_dot[kMaxSpinCompress * 3] = {};
  double raw1_dot[kMaxSpinCompress * 9] = {};

  const int type1 = types[atom];
  const int type_pair_base = type1 * num_types;
  const int basis_count = spin_basis_size + 1;
  const double rcinv = 1.0 / static_cast<double>(spin_cutoff);
  const int radial_count = nn_radial[atom];

  for (int c = 0; c < spin_compress; ++c) {
    for (int k = 0; k < 3; ++k) {
      rho0[idx2(c, k, 3)] =
          density_rho0_cache[atom + atom_stride * (c * 3 + k)];
      rho0_dot[idx2(c, k, 3)] =
          density_rho0_dot_cache[atom + atom_stride * (c * 3 + k)];
    }
    for (int k = 0; k < 9; ++k) {
      raw1[idx2(c, k, 9)] =
          density_raw1_cache[atom + atom_stride * (c * 9 + k)];
      l1_stf[idx2(c, k, 9)] =
          density_l1_stf_cache[atom + atom_stride * (c * 9 + k)];
      geom[idx2(c, k, 9)] =
          density_geom_cache[atom + atom_stride * (c * 9 + k)];
      raw1_dot[idx2(c, k, 9)] =
          density_raw1_dot_cache[atom + atom_stride * (c * 9 + k)];
    }
    l1_rdot[c] = density_l1_rdot_cache[atom + atom_stride * c];
    for (int k = 0; k < 3; ++k) {
      l1_cross[idx2(c, k, 3)] =
          density_l1_cross_cache[atom + atom_stride * (c * 3 + k)];
    }
    if (spin_l_max >= 2) {
      for (int k = 0; k < 15; ++k) {
        angular2[idx2(c, k, 15)] =
            density_angular2_cache[atom + atom_stride * (c * 15 + k)];
      }
    }
    if (spin_l_max >= 3) {
      for (int k = 0; k < 21; ++k) {
        angular3[idx2(c, k, 21)] =
            density_angular3_cache[atom + atom_stride * (c * 21 + k)];
      }
    }
    if (spin_l_max >= 4) {
      for (int k = 0; k < 27; ++k) {
        angular4[idx2(c, k, 27)] =
            density_angular4_cache[atom + atom_stride * (c * 27 + k)];
      }
    }
  }

  const int rho0_offset = 2 + 4 * spin_compress;
  const int l1_rdot_offset = rho0_offset + spin_compress;
  const int l1_cross_offset = l1_rdot_offset + spin_compress;
  const int l1_stf_offset = l1_cross_offset + spin_compress;
  int angular_offset = rho0_offset + spin_compress;
  if (spin_l_max >= 1) {
    angular_offset += 3 * spin_compress;
  }
  int geom_offset = angular_offset;
  for (int ell = 2; ell <= spin_l_max; ++ell) {
    geom_offset += spin_compress;
  }
  const int rho0_dot_offset = geom_offset + spin_compress;
  const int raw1_dot_offset = rho0_dot_offset + spin_compress;

  const double si0[3] = {
      spins_soa3[atom],
      spins_soa3[atom_stride + atom],
      spins_soa3[2 * atom_stride + atom]};

  double rho0_pull[kMaxSpinCompress * 3] = {};
  double rho0_dot_pull[kMaxSpinCompress * 3] = {};
  double l1_pull[kMaxSpinCompress * 9] = {};
  double l1_dot_pull[kMaxSpinCompress * 9] = {};
  double geom_pull[kMaxSpinCompress * 9] = {};
  double grad_spin_i_direct[3] = {0.0, 0.0, 0.0};

  for (int c = 0; c < spin_compress; ++c) {
    const double alpha0 =
        static_cast<double>(fp[atom + atom_stride * (struct_dim + rho0_offset + c)]);
    const double alpha0_dot =
        static_cast<double>(fp[atom + atom_stride * (struct_dim + rho0_dot_offset + c)]);
    for (int d = 0; d < 3; ++d) {
      rho0_pull[idx2(c, d, 3)] =
          2.0 * alpha0 * rho0[idx2(c, d, 3)] +
          alpha0_dot * rho0_dot[idx2(c, d, 3)];
      rho0_dot_pull[idx2(c, d, 3)] = alpha0_dot * rho0[idx2(c, d, 3)];
    }

    const double alpha_geom =
        static_cast<double>(fp[atom + atom_stride * (struct_dim + geom_offset + c)]);
    const double* g = geom + c * 9;
    for (int a = 0; a < 3; ++a) {
      double gs = 0.0;
      for (int b = 0; b < 3; ++b) {
        gs += (g[3 * a + b] + g[3 * b + a]) * si0[b];
      }
      grad_spin_i_direct[a] += alpha_geom * gs;
    }
    double ss[9];
    stf_outer3(si0, si0, ss);
    for (int k = 0; k < 9; ++k) {
      geom_pull[c * 9 + k] = alpha_geom * ss[k];
    }
  }

  if (spin_l_max >= 1) {
    for (int c = 0; c < spin_compress; ++c) {
      double* mat = l1_pull + c * 9;
      double* mat_dot = l1_dot_pull + c * 9;
      const double alpha_rdot =
          static_cast<double>(fp[atom + atom_stride * (struct_dim + l1_rdot_offset + c)]);
      const double alpha_cross =
          static_cast<double>(fp[atom + atom_stride * (struct_dim + l1_cross_offset + c)]);
      const double alpha_stf =
          static_cast<double>(fp[atom + atom_stride * (struct_dim + l1_stf_offset + c)]);
      const double alpha_raw1 =
          static_cast<double>(fp[atom + atom_stride * (struct_dim + raw1_dot_offset + c)]);
      const double rdot = l1_rdot[c];
      const double* cross = l1_cross + c * 3;
      const double* stf = l1_stf + c * 9;
      const double* raw = raw1 + c * 9;
      const double* raw_dot = raw1_dot + c * 9;
      const double g_cross[3] = {
          2.0 * alpha_cross * cross[0],
          2.0 * alpha_cross * cross[1],
          2.0 * alpha_cross * cross[2]};
      mat[0] += 2.0 * alpha_rdot * rdot;
      mat[4] += 2.0 * alpha_rdot * rdot;
      mat[8] += 2.0 * alpha_rdot * rdot;
      mat[1] += g_cross[2];
      mat[2] -= g_cross[1];
      mat[3] -= g_cross[2];
      mat[5] += g_cross[0];
      mat[6] += g_cross[1];
      mat[7] -= g_cross[0];
      for (int k = 0; k < 9; ++k) {
        mat[k] += 2.0 * alpha_stf * stf[k] + alpha_raw1 * raw_dot[k];
        mat_dot[k] = alpha_raw1 * raw[k];
      }
    }
  }

  for (int d = 0; d < 3; ++d) {
    atomicAdd(mforce_soa3 + d * atom_stride + atom, -grad_spin_i_direct[d]);
  }

  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    double rhat[3];
    double dist = 0.0;
    double si[3];
    double sj[3];
    if (use_cached_geometry) {
      load_spin_edge_cached(
          atom,
          neighbor,
          atom_stride,
          slot,
          spins_soa3,
          spin_edge_dx,
          spin_edge_dy,
          spin_edge_dz,
          spin_edge_dist,
          rhat,
          dist,
          si,
          sj);
    } else {
      load_spin_edge(atom, neighbor, atom_stride, box, positions_soa3, spins_soa3,
                     rhat, dist, si, sj);
    }
    if (dist <= 1.0e-12 || dist >= spin_cutoff) {
      continue;
    }
    double weights[kMaxSpinCompress] = {};
    double weight_derivatives[kMaxSpinCompress] = {};
    if (spin_edge_weights != nullptr && spin_edge_weight_derivatives != nullptr &&
        spin_compress == 4 && spin_basis_size == 3) {
      for (int c = 0; c < 4; ++c) {
        const int cache_index = spin_edge_cache_index(atom_stride, slot, atom, c);
        weights[c] = static_cast<double>(spin_edge_weights[cache_index]);
        weight_derivatives[c] =
            static_cast<double>(spin_edge_weight_derivatives[cache_index]);
      }
    } else {
      const int type_pair = type_pair_base + types[neighbor];
      double fc = 0.0;
      double fcp = 0.0;
      double fn[kMaxSpinBasis] = {};
      double fnp[kMaxSpinBasis] = {};
      find_fc_and_fcp(spin_cutoff, rcinv, dist, fc, fcp);
      find_fn_and_fnp(spin_basis_size, rcinv, dist, fc, fcp, fn, fnp);
      for (int c = 0; c < spin_compress; ++c) {
        for (int k = 0; k < basis_count; ++k) {
          const int index = spin_coefficient_offset +
              ((c * basis_count + k) * num_types * num_types + type_pair);
          const double coefficient =
              static_cast<double>(descriptor_coefficients[index]);
          weights[c] += fn[k] * coefficient;
          weight_derivatives[c] += fnp[k] * coefficient;
        }
      }
    }

    const double dot = dot3(si, sj);
    double grad_weight[kMaxSpinCompress] = {};
    double grad_rhat[3] = {0.0, 0.0, 0.0};
    double grad_si[3] = {0.0, 0.0, 0.0};
    double grad_sj[3] = {0.0, 0.0, 0.0};
    double grad_dot = 0.0;

    for (int c = 0; c < spin_compress; ++c) {
      const double* b = rho0_pull + c * 3;
      const double* bd = rho0_dot_pull + c * 3;
      const double u[3] = {
          b[0] + dot * bd[0],
          b[1] + dot * bd[1],
          b[2] + dot * bd[2]};
      const double w = weights[c];
      grad_weight[c] += dot3(sj, u);
      for (int d = 0; d < 3; ++d) {
        grad_sj[d] += w * u[d];
      }
      grad_dot += w * dot3(sj, bd);
    }

    for (int c = 0; c < spin_compress; ++c) {
      const double* k_mat = geom_pull + c * 9;
      const double kr[3] = {
          k_mat[0] * rhat[0] + k_mat[1] * rhat[1] + k_mat[2] * rhat[2],
          k_mat[3] * rhat[0] + k_mat[4] * rhat[1] + k_mat[5] * rhat[2],
          k_mat[6] * rhat[0] + k_mat[7] * rhat[1] + k_mat[8] * rhat[2]};
      double u[3] = {0.0, 0.0, 0.0};
      double mt[3] = {0.0, 0.0, 0.0};
      double dot_v2 = 0.0;
      if (spin_l_max >= 1) {
        const double* m = l1_pull + c * 9;
        const double* md = l1_dot_pull + c * 9;
        const double v1[3] = {
            m[0] * sj[0] + m[1] * sj[1] + m[2] * sj[2],
            m[3] * sj[0] + m[4] * sj[1] + m[5] * sj[2],
            m[6] * sj[0] + m[7] * sj[1] + m[8] * sj[2]};
        const double v2[3] = {
            md[0] * sj[0] + md[1] * sj[1] + md[2] * sj[2],
            md[3] * sj[0] + md[4] * sj[1] + md[5] * sj[2],
            md[6] * sj[0] + md[7] * sj[1] + md[8] * sj[2]};
        for (int d = 0; d < 3; ++d) {
          u[d] = v1[d] + dot * v2[d];
        }
        mt[0] = m[0] * rhat[0] + m[3] * rhat[1] + m[6] * rhat[2] +
                dot * (md[0] * rhat[0] + md[3] * rhat[1] + md[6] * rhat[2]);
        mt[1] = m[1] * rhat[0] + m[4] * rhat[1] + m[7] * rhat[2] +
                dot * (md[1] * rhat[0] + md[4] * rhat[1] + md[7] * rhat[2]);
        mt[2] = m[2] * rhat[0] + m[5] * rhat[1] + m[8] * rhat[2] +
                dot * (md[2] * rhat[0] + md[5] * rhat[1] + md[8] * rhat[2]);
        dot_v2 = dot3(rhat, v2);
      }
      const double w = weights[c];
      for (int d = 0; d < 3; ++d) {
        grad_weight[c] += rhat[d] * (u[d] + kr[d]);
        grad_rhat[d] += w * (u[d] + 2.0 * kr[d]);
        grad_sj[d] += w * mt[d];
      }
      grad_dot += w * dot_v2;
    }

    int offset = angular_offset;
    for (int ell = 2; ell <= spin_l_max; ++ell) {
      const int width = (2 * ell + 1) * 3;
      const double* angular = ell == 2 ? angular2 : ell == 3 ? angular3 : angular4;
      double ylm[9];
      const int ylm_width = real_spherical_harmonics_spin(rhat, ell, ylm);
      double value[27];
      int value_width = 0;
      for (int m = 0; m < ylm_width; ++m) {
        value[value_width++] = ylm[m] * sj[0];
        value[value_width++] = ylm[m] * sj[1];
        value[value_width++] = ylm[m] * sj[2];
      }
      double ge[27] = {};
      for (int c = 0; c < spin_compress; ++c) {
        const double alpha =
            static_cast<double>(fp[atom + atom_stride * (struct_dim + offset + c)]);
        const double* self = angular + c * width;
        for (int k = 0; k < width; ++k) {
          const double gd = 2.0 * alpha * self[k];
          grad_weight[c] += gd * value[k];
          ge[k] += gd * weights[c];
        }
      }
      double grad_ylm[9] = {};
      for (int m = 0; m < ylm_width; ++m) {
        for (int d = 0; d < 3; ++d) {
          const double g = ge[m * 3 + d];
          grad_sj[d] += g * ylm[m];
          grad_ylm[m] += g * sj[d];
        }
      }
      add_real_spherical_harmonics_gradient(rhat, ell, grad_ylm, grad_rhat);
      offset += spin_compress;
    }

    for (int d = 0; d < 3; ++d) {
      grad_si[d] += grad_dot * sj[d];
      grad_sj[d] += grad_dot * si[d];
    }

    double grad_dist = 0.0;
    for (int c = 0; c < spin_compress; ++c) {
      grad_dist += grad_weight[c] * weight_derivatives[c];
    }
    double dot_r = 0.0;
    for (int d = 0; d < 3; ++d) {
      dot_r += grad_rhat[d] * rhat[d];
    }
    double grad_rij[3];
    for (int d = 0; d < 3; ++d) {
      grad_rij[d] = grad_dist * rhat[d] +
                    (grad_rhat[d] - dot_r * rhat[d]) / dist;
      atomicAdd(force_soa3 + d * atom_stride + atom, grad_rij[d]);
      atomicAdd(force_soa3 + d * atom_stride + neighbor, -grad_rij[d]);
      atomicAdd(mforce_soa3 + d * atom_stride + atom, -grad_si[d]);
      atomicAdd(mforce_soa3 + d * atom_stride + neighbor, -grad_sj[d]);
    }
    if (accumulate_virial) {
      for (int a = 0; a < 3; ++a) {
        const double rij_a = rhat[a] * dist;
        for (int b = 0; b < 3; ++b) {
          atomicAdd(
              virial_soa9 + virial_internal_component(a * 3 + b) * atom_stride + atom,
              -rij_a * grad_rij[b]);
        }
      }
    }
  }
}

template <bool AtomMajor>
__global__ void prepare_spin_density_pulls_c4_l4(
    int atom_count,
    int atom_stride,
    int struct_dim,
    const double* __restrict__ spins_soa3,
    const float* __restrict__ fp,
    float* __restrict__ density_rho0_cache,
    const float* __restrict__ density_l1_rdot_cache,
    const float* __restrict__ density_l1_cross_cache,
    float* __restrict__ density_l1_stf_cache,
    float* __restrict__ density_angular2_cache,
    float* __restrict__ density_angular3_cache,
    float* __restrict__ density_angular4_cache,
    const float* __restrict__ density_geom_cache,
    float* __restrict__ density_rho0_dot_cache,
    float* __restrict__ density_raw1_cache,
    float* __restrict__ density_raw1_dot_cache,
    double* __restrict__ mforce_soa3) {
  constexpr int C = 4;
  constexpr int Rho0Offset = 18;
  constexpr int L1RdotOffset = 22;
  constexpr int L1CrossOffset = 26;
  constexpr int L1StfOffset = 30;
  constexpr int Angular2Offset = 34;
  constexpr int Angular3Offset = 38;
  constexpr int Angular4Offset = 42;
  constexpr int GeomOffset = 46;
  constexpr int Rho0DotOffset = 50;
  constexpr int Raw1DotOffset = 54;
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const double si0[3] = {
      spins_soa3[atom],
      spins_soa3[atom_stride + atom],
      spins_soa3[2 * atom_stride + atom]};
  double grad_spin_i_direct[3] = {0.0, 0.0, 0.0};
  double ss[9];
  stf_outer3(si0, si0, ss);

  for (int c = 0; c < C; ++c) {
    const double alpha0 = static_cast<double>(
        fp[atom + atom_stride * (struct_dim + Rho0Offset + c)]);
    const double alpha0_dot = static_cast<double>(
        fp[atom + atom_stride * (struct_dim + Rho0DotOffset + c)]);
    for (int d = 0; d < 3; ++d) {
      const int idx = spin_component_cache_index<AtomMajor, C * 3>(
          atom_stride, atom, c * 3 + d);
      const double rho0 = static_cast<double>(density_rho0_cache[idx]);
      const double rho0_dot = static_cast<double>(density_rho0_dot_cache[idx]);
      density_rho0_cache[idx] =
          static_cast<float>(2.0 * alpha0 * rho0 + alpha0_dot * rho0_dot);
      density_rho0_dot_cache[idx] = static_cast<float>(alpha0_dot * rho0);
    }

    const double alpha_geom = static_cast<double>(
        fp[atom + atom_stride * (struct_dim + GeomOffset + c)]);
    double geom[9];
    for (int k = 0; k < 9; ++k) {
      geom[k] = static_cast<double>(
          density_geom_cache[
              spin_component_cache_index<AtomMajor, C * 9>(
                  atom_stride, atom, c * 9 + k)]);
    }
    for (int a = 0; a < 3; ++a) {
      double gs = 0.0;
      for (int b = 0; b < 3; ++b) {
        gs += (geom[3 * a + b] + geom[3 * b + a]) * si0[b];
      }
      grad_spin_i_direct[a] += alpha_geom * gs;
    }

    const double alpha_rdot = static_cast<double>(
        fp[atom + atom_stride * (struct_dim + L1RdotOffset + c)]);
    const double alpha_cross = static_cast<double>(
        fp[atom + atom_stride * (struct_dim + L1CrossOffset + c)]);
    const double alpha_stf = static_cast<double>(
        fp[atom + atom_stride * (struct_dim + L1StfOffset + c)]);
    const double alpha_raw1 = static_cast<double>(
        fp[atom + atom_stride * (struct_dim + Raw1DotOffset + c)]);
    double mat[9] = {};
    const double rdot = static_cast<double>(
        density_l1_rdot_cache[
            spin_component_cache_index<AtomMajor, C>(atom_stride, atom, c)]);
    const double cross[3] = {
        static_cast<double>(density_l1_cross_cache[
            spin_component_cache_index<AtomMajor, C * 3>(
                atom_stride, atom, c * 3)]),
        static_cast<double>(
            density_l1_cross_cache[
                spin_component_cache_index<AtomMajor, C * 3>(
                    atom_stride, atom, c * 3 + 1)]),
        static_cast<double>(
            density_l1_cross_cache[
                spin_component_cache_index<AtomMajor, C * 3>(
                    atom_stride, atom, c * 3 + 2)])};
    const double g_cross[3] = {
        2.0 * alpha_cross * cross[0],
        2.0 * alpha_cross * cross[1],
        2.0 * alpha_cross * cross[2]};
    mat[0] += 2.0 * alpha_rdot * rdot;
    mat[4] += 2.0 * alpha_rdot * rdot;
    mat[8] += 2.0 * alpha_rdot * rdot;
    mat[1] += g_cross[2];
    mat[2] -= g_cross[1];
    mat[3] -= g_cross[2];
    mat[5] += g_cross[0];
    mat[6] += g_cross[1];
    mat[7] -= g_cross[0];
    for (int k = 0; k < 9; ++k) {
      const int idx = spin_component_cache_index<AtomMajor, C * 9>(
          atom_stride, atom, c * 9 + k);
      const double stf = static_cast<double>(density_l1_stf_cache[idx]);
      const double raw = static_cast<double>(density_raw1_cache[idx]);
      const double raw_dot = static_cast<double>(density_raw1_dot_cache[idx]);
      density_l1_stf_cache[idx] =
          static_cast<float>(mat[k] + 2.0 * alpha_stf * stf +
                             alpha_raw1 * raw_dot);
      density_raw1_dot_cache[idx] = static_cast<float>(alpha_raw1 * raw);
      density_raw1_cache[idx] = static_cast<float>(alpha_geom * ss[k]);
    }

    const double alpha_l2 = static_cast<double>(
        fp[atom + atom_stride * (struct_dim + Angular2Offset + c)]);
    for (int k = 0; k < 15; ++k) {
      const int idx = spin_component_cache_index<AtomMajor, C * 15>(
          atom_stride, atom, c * 15 + k);
      density_angular2_cache[idx] =
          static_cast<float>(2.0 * alpha_l2 *
                             static_cast<double>(density_angular2_cache[idx]));
    }
    const double alpha_l3 = static_cast<double>(
        fp[atom + atom_stride * (struct_dim + Angular3Offset + c)]);
    for (int k = 0; k < 21; ++k) {
      const int idx = spin_component_cache_index<AtomMajor, C * 21>(
          atom_stride, atom, c * 21 + k);
      density_angular3_cache[idx] =
          static_cast<float>(2.0 * alpha_l3 *
                             static_cast<double>(density_angular3_cache[idx]));
    }
    const double alpha_l4 = static_cast<double>(
        fp[atom + atom_stride * (struct_dim + Angular4Offset + c)]);
    for (int k = 0; k < 27; ++k) {
      const int idx = spin_component_cache_index<AtomMajor, C * 27>(
          atom_stride, atom, c * 27 + k);
      density_angular4_cache[idx] =
          static_cast<float>(2.0 * alpha_l4 *
                             static_cast<double>(density_angular4_cache[idx]));
    }
  }

  for (int d = 0; d < 3; ++d) {
    mforce_soa3[d * atom_stride + atom] -= grad_spin_i_direct[d];
  }
}

template <bool AtomMajor>
__global__ void __launch_bounds__(32, 12) accumulate_spin_density_forces_c4_l4_pull(
    int atom_count,
    int atom_stride,
    int struct_dim,
    float spin_cutoff,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ fp,
    const float* __restrict__ spin_edge_dx,
    const float* __restrict__ spin_edge_dy,
    const float* __restrict__ spin_edge_dz,
    const float* __restrict__ spin_edge_dist,
    const float* __restrict__ spin_edge_weights,
    const float* __restrict__ spin_edge_weight_derivatives,
    const float* __restrict__ density_rho0_pull_cache,
    const float* __restrict__ density_l1_pull_cache,
    const float* __restrict__ density_angular2_cache,
    const float* __restrict__ density_angular3_cache,
    const float* __restrict__ density_angular4_cache,
    const float* __restrict__ density_geom_pull_cache,
    const float* __restrict__ density_rho0_dot_pull_cache,
    const float* __restrict__ density_l1_dot_pull_cache,
    double* __restrict__ force_soa3,
    double* __restrict__ mforce_soa3,
    bool accumulate_virial,
    double* __restrict__ virial_soa9) {
  constexpr int C = 4;
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  const int cache_stride = spin_component_cache_stride<AtomMajor>(atom_stride);

  const int radial_count = nn_radial[atom];
  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    double rhat[3];
    double dist = 0.0;
    double si[3];
    double sj[3];
    load_spin_edge_cached(
        atom,
        neighbor,
        atom_stride,
        slot,
        spins_soa3,
        spin_edge_dx,
        spin_edge_dy,
        spin_edge_dz,
        spin_edge_dist,
        rhat,
        dist,
        si,
        sj);
    if (dist <= 1.0e-12 || dist >= spin_cutoff) {
      continue;
    }

    double weights[C];
    double weight_derivatives[C];
    load_spin_edge_weight_derivative_cache(
        atom_stride,
        slot,
        atom,
        spin_edge_weights,
        spin_edge_weight_derivatives,
        weights,
        weight_derivatives);

    const double dot = dot3(si, sj);
    double grad_weight[C] = {};
    double grad_rhat[3] = {0.0, 0.0, 0.0};
    double grad_si[3] = {0.0, 0.0, 0.0};
    double grad_sj[3] = {0.0, 0.0, 0.0};
    double grad_dot = 0.0;

    for (int c = 0; c < C; ++c) {
      const double b[3] = {
          static_cast<double>(
              density_rho0_pull_cache[
                  spin_component_cache_index<AtomMajor, C * 3>(
                      atom_stride, atom, c * 3)]),
          static_cast<double>(
              density_rho0_pull_cache[
                  spin_component_cache_index<AtomMajor, C * 3>(
                      atom_stride, atom, c * 3 + 1)]),
          static_cast<double>(
              density_rho0_pull_cache[
                  spin_component_cache_index<AtomMajor, C * 3>(
                      atom_stride, atom, c * 3 + 2)])};
      const double bd[3] = {
          static_cast<double>(
              density_rho0_dot_pull_cache[
                  spin_component_cache_index<AtomMajor, C * 3>(
                      atom_stride, atom, c * 3)]),
          static_cast<double>(
              density_rho0_dot_pull_cache[
                  spin_component_cache_index<AtomMajor, C * 3>(
                      atom_stride, atom, c * 3 + 1)]),
          static_cast<double>(
              density_rho0_dot_pull_cache[
                  spin_component_cache_index<AtomMajor, C * 3>(
                      atom_stride, atom, c * 3 + 2)])};
      const double u[3] = {
          b[0] + dot * bd[0],
          b[1] + dot * bd[1],
          b[2] + dot * bd[2]};
      const double w = weights[c];
      grad_weight[c] += dot3(sj, u);
      for (int d = 0; d < 3; ++d) {
        grad_sj[d] += w * u[d];
      }
      grad_dot += w * dot3(sj, bd);

      const float* gbase = density_geom_pull_cache +
          spin_component_cache_index<AtomMajor, C * 9>(
              atom_stride, atom, c * 9);
      const float* mbase = density_l1_pull_cache +
          spin_component_cache_index<AtomMajor, C * 9>(
              atom_stride, atom, c * 9);
      const float* mdbase =
          density_l1_dot_pull_cache +
          spin_component_cache_index<AtomMajor, C * 9>(
              atom_stride, atom, c * 9);
      const double kr[3] = {
          static_cast<double>(gbase[0 * cache_stride]) * rhat[0] +
              static_cast<double>(gbase[1 * cache_stride]) * rhat[1] +
              static_cast<double>(gbase[2 * cache_stride]) * rhat[2],
          static_cast<double>(gbase[3 * cache_stride]) * rhat[0] +
              static_cast<double>(gbase[4 * cache_stride]) * rhat[1] +
              static_cast<double>(gbase[5 * cache_stride]) * rhat[2],
          static_cast<double>(gbase[6 * cache_stride]) * rhat[0] +
              static_cast<double>(gbase[7 * cache_stride]) * rhat[1] +
              static_cast<double>(gbase[8 * cache_stride]) * rhat[2]};
      const double v1[3] = {
          static_cast<double>(mbase[0 * cache_stride]) * sj[0] +
              static_cast<double>(mbase[1 * cache_stride]) * sj[1] +
              static_cast<double>(mbase[2 * cache_stride]) * sj[2],
          static_cast<double>(mbase[3 * cache_stride]) * sj[0] +
              static_cast<double>(mbase[4 * cache_stride]) * sj[1] +
              static_cast<double>(mbase[5 * cache_stride]) * sj[2],
          static_cast<double>(mbase[6 * cache_stride]) * sj[0] +
              static_cast<double>(mbase[7 * cache_stride]) * sj[1] +
              static_cast<double>(mbase[8 * cache_stride]) * sj[2]};
      const double v2[3] = {
          static_cast<double>(mdbase[0 * cache_stride]) * sj[0] +
              static_cast<double>(mdbase[1 * cache_stride]) * sj[1] +
              static_cast<double>(mdbase[2 * cache_stride]) * sj[2],
          static_cast<double>(mdbase[3 * cache_stride]) * sj[0] +
              static_cast<double>(mdbase[4 * cache_stride]) * sj[1] +
              static_cast<double>(mdbase[5 * cache_stride]) * sj[2],
          static_cast<double>(mdbase[6 * cache_stride]) * sj[0] +
              static_cast<double>(mdbase[7 * cache_stride]) * sj[1] +
              static_cast<double>(mdbase[8 * cache_stride]) * sj[2]};
      const double u1[3] = {
          v1[0] + dot * v2[0],
          v1[1] + dot * v2[1],
          v1[2] + dot * v2[2]};
      const double mt[3] = {
          static_cast<double>(mbase[0 * cache_stride]) * rhat[0] +
              static_cast<double>(mbase[3 * cache_stride]) * rhat[1] +
              static_cast<double>(mbase[6 * cache_stride]) * rhat[2] +
              dot * (static_cast<double>(mdbase[0 * cache_stride]) * rhat[0] +
                     static_cast<double>(mdbase[3 * cache_stride]) * rhat[1] +
                     static_cast<double>(mdbase[6 * cache_stride]) * rhat[2]),
          static_cast<double>(mbase[1 * cache_stride]) * rhat[0] +
              static_cast<double>(mbase[4 * cache_stride]) * rhat[1] +
              static_cast<double>(mbase[7 * cache_stride]) * rhat[2] +
              dot * (static_cast<double>(mdbase[1 * cache_stride]) * rhat[0] +
                     static_cast<double>(mdbase[4 * cache_stride]) * rhat[1] +
                     static_cast<double>(mdbase[7 * cache_stride]) * rhat[2]),
          static_cast<double>(mbase[2 * cache_stride]) * rhat[0] +
              static_cast<double>(mbase[5 * cache_stride]) * rhat[1] +
              static_cast<double>(mbase[8 * cache_stride]) * rhat[2] +
              dot * (static_cast<double>(mdbase[2 * cache_stride]) * rhat[0] +
                     static_cast<double>(mdbase[5 * cache_stride]) * rhat[1] +
                     static_cast<double>(mdbase[8 * cache_stride]) * rhat[2])};
      for (int d = 0; d < 3; ++d) {
        grad_weight[c] += rhat[d] * (u1[d] + kr[d]);
        grad_rhat[d] += w * (u1[d] + 2.0 * kr[d]);
        grad_sj[d] += w * mt[d];
      }
      grad_dot += w * dot3(rhat, v2);
    }

    for (int ell = 2; ell <= 4; ++ell) {
      const int width = (2 * ell + 1) * 3;
      const float* angular =
          ell == 2 ? density_angular2_cache :
          ell == 3 ? density_angular3_cache : density_angular4_cache;
      double ylm[9];
      const int ylm_width = real_spherical_harmonics_spin(rhat, ell, ylm);
      double ge[27] = {};
      for (int c = 0; c < C; ++c) {
        for (int m = 0; m < ylm_width; ++m) {
          for (int d = 0; d < 3; ++d) {
            const int k = m * 3 + d;
            const double gd = static_cast<double>(
                angular[spin_component_cache_index<AtomMajor>(
                    atom_stride, C * width, atom, c * width + k)]);
            grad_weight[c] += gd * ylm[m] * sj[d];
            ge[k] += gd * weights[c];
          }
        }
      }
      double grad_ylm[9] = {};
      for (int m = 0; m < ylm_width; ++m) {
        for (int d = 0; d < 3; ++d) {
          const double g = ge[m * 3 + d];
          grad_sj[d] += g * ylm[m];
          grad_ylm[m] += g * sj[d];
        }
      }
      add_real_spherical_harmonics_gradient(rhat, ell, grad_ylm, grad_rhat);
    }

    for (int d = 0; d < 3; ++d) {
      grad_si[d] += grad_dot * sj[d];
      grad_sj[d] += grad_dot * si[d];
    }

    double grad_dist = 0.0;
    for (int c = 0; c < C; ++c) {
      grad_dist += grad_weight[c] * weight_derivatives[c];
    }
    double dot_r = 0.0;
    for (int d = 0; d < 3; ++d) {
      dot_r += grad_rhat[d] * rhat[d];
    }
    double grad_rij[3];
    for (int d = 0; d < 3; ++d) {
      grad_rij[d] = grad_dist * rhat[d] +
                    (grad_rhat[d] - dot_r * rhat[d]) / dist;
      atomicAdd(force_soa3 + d * atom_stride + atom, grad_rij[d]);
      atomicAdd(force_soa3 + d * atom_stride + neighbor, -grad_rij[d]);
      atomicAdd(mforce_soa3 + d * atom_stride + atom, -grad_si[d]);
      atomicAdd(mforce_soa3 + d * atom_stride + neighbor, -grad_sj[d]);
    }
    if (accumulate_virial) {
      for (int a = 0; a < 3; ++a) {
        const double rij_a = rhat[a] * dist;
        for (int b = 0; b < 3; ++b) {
          atomicAdd(
              virial_soa9 + virial_internal_component(a * 3 + b) *
                                 atom_stride + atom,
              -rij_a * grad_rij[b]);
        }
      }
    }
  }
}

__device__ __forceinline__ int spin_dim_without_chiral(int c_count, int l_max) {
  int dim = 2 + 4 * c_count;
  if (l_max >= 0) {
    dim += c_count;
  }
  if (l_max >= 1) {
    dim += 3 * c_count;
  }
  for (int ell = 2; ell <= l_max; ++ell) {
    dim += c_count;
  }
  dim += c_count;
  dim += c_count;
  if (l_max >= 1) {
    dim += c_count;
  }
  return dim;
}

__device__ void apply_edge_gradients(
    int atom,
    int neighbor,
    int atom_stride,
    double dist,
    const double* rhat,
    const double* si,
    const double* sj,
    double grad_weight,
    double weight_derivative,
    const double* grad_rhat,
    const double* grad_si,
    const double* grad_sj,
    double* force_soa3,
    double* mforce_soa3,
    bool accumulate_virial,
    double* virial_soa9) {
  (void)si;
  (void)sj;
  const double grad_dist = grad_weight * weight_derivative;
  double dot_r = 0.0;
  for (int d = 0; d < 3; ++d) {
    dot_r += grad_rhat[d] * rhat[d];
  }
  double grad_rij[3];
  for (int d = 0; d < 3; ++d) {
    grad_rij[d] =
        grad_dist * rhat[d] + (grad_rhat[d] - dot_r * rhat[d]) / dist;
    atomicAdd(force_soa3 + d * atom_stride + atom, grad_rij[d]);
    atomicAdd(force_soa3 + d * atom_stride + neighbor, -grad_rij[d]);
    atomicAdd(mforce_soa3 + d * atom_stride + atom, -grad_si[d]);
    atomicAdd(mforce_soa3 + d * atom_stride + neighbor, -grad_sj[d]);
  }
  if (accumulate_virial) {
    for (int a = 0; a < 3; ++a) {
      const double rij_a = rhat[a] * dist;
      for (int b = 0; b < 3; ++b) {
        atomicAdd(
            virial_soa9 + virial_internal_component(a * 3 + b) * atom_stride + atom,
            -rij_a * grad_rij[b]);
      }
    }
  }
}

__device__ __forceinline__ void cross3f(
    const float* a,
    const float* b,
    float* out) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ __forceinline__ float dot3f(const float* a, const float* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ __forceinline__ void stf_outer3f(
    const float* a,
    const float* b,
    float* out) {
  const float trace = dot3f(a, b) / 3.0f;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      float value = 0.5f * (a[i] * b[j] + a[j] * b[i]);
      if (i == j) {
        value -= trace;
      }
      out[3 * i + j] = value;
    }
  }
}

__device__ __forceinline__ void load_spin_edge_cached_f32(
    int atom,
    int neighbor,
    int atom_stride,
    int slot,
    const double* spins_soa3,
    const float* spin_edge_dx,
    const float* spin_edge_dy,
    const float* spin_edge_dz,
    const float* spin_edge_dist,
    float* rhat,
    float& dist,
    float* si,
    float* sj) {
  const int offset = atom + atom_stride * slot;
  const float dx = spin_edge_dx[offset];
  const float dy = spin_edge_dy[offset];
  const float dz = spin_edge_dz[offset];
  dist = spin_edge_dist[offset];
  if (dist > 0.0f) {
    const float inv_dist = 1.0f / dist;
    rhat[0] = dx * inv_dist;
    rhat[1] = dy * inv_dist;
    rhat[2] = dz * inv_dist;
  } else {
    rhat[0] = 0.0f;
    rhat[1] = 0.0f;
    rhat[2] = 0.0f;
  }
  si[0] = static_cast<float>(spins_soa3[atom]);
  si[1] = static_cast<float>(spins_soa3[atom_stride + atom]);
  si[2] = static_cast<float>(spins_soa3[2 * atom_stride + atom]);
  sj[0] = static_cast<float>(spins_soa3[neighbor]);
  sj[1] = static_cast<float>(spins_soa3[atom_stride + neighbor]);
  sj[2] = static_cast<float>(spins_soa3[2 * atom_stride + neighbor]);
}

__device__ __forceinline__ void load_spin_edge_weight_cache_f32(
    int atom_stride,
    int slot,
    int atom,
    const float* __restrict__ spin_edge_weights,
    float* weights) {
  for (int c = 0; c < 4; ++c) {
    weights[c] = spin_edge_weights[spin_edge_cache_index(atom_stride, slot, atom, c)];
  }
}

__device__ __forceinline__ void load_spin_edge_weight_derivative_cache_f32(
    int atom_stride,
    int slot,
    int atom,
    const float* __restrict__ spin_edge_weights,
    const float* __restrict__ spin_edge_weight_derivatives,
    float* weights,
    float* weight_derivatives) {
  for (int c = 0; c < 4; ++c) {
    const int index = spin_edge_cache_index(atom_stride, slot, atom, c);
    weights[c] = spin_edge_weights[index];
    weight_derivatives[c] = spin_edge_weight_derivatives[index];
  }
}

__device__ __forceinline__ void fill_spin_monomials2f(
    const float* u,
    float* m2) {
  const float x = u[0];
  const float y = u[1];
  const float z = u[2];
  m2[0] = x * x;
  m2[1] = y * y;
  m2[2] = z * z;
  m2[3] = x * y;
  m2[4] = x * z;
  m2[5] = y * z;
}

__device__ __forceinline__ void fill_spin_monomialsf(
    const float* u,
    float* m3,
    float* m4) {
  const float x = u[0];
  const float y = u[1];
  const float z = u[2];
  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  const float x3 = x2 * x;
  const float y3 = y2 * y;
  const float z3 = z2 * z;
  m3[0] = x3;
  m3[1] = y3;
  m3[2] = z3;
  m3[3] = x2 * y;
  m3[4] = x2 * z;
  m3[5] = x * y2;
  m3[6] = y2 * z;
  m3[7] = x * z2;
  m3[8] = y * z2;
  m3[9] = x * y * z;
  m4[0] = x2 * x2;
  m4[1] = y2 * y2;
  m4[2] = z2 * z2;
  m4[3] = x3 * y;
  m4[4] = x3 * z;
  m4[5] = x * y3;
  m4[6] = y3 * z;
  m4[7] = x * z3;
  m4[8] = y * z3;
  m4[9] = x2 * y2;
  m4[10] = x2 * z2;
  m4[11] = y2 * z2;
  m4[12] = x2 * y * z;
  m4[13] = x * y2 * z;
  m4[14] = x * y * z2;
}

__device__ __forceinline__ float dot_spin_termsf(
    const float* lhs,
    const float* rhs,
    int count) {
  float out = 0.0f;
  for (int k = 0; k < count; ++k) {
    out += lhs[k] * rhs[k];
  }
  return out;
}

__device__ __forceinline__ void project_rank2_spin_gradientf(
    const float* grad,
    float* terms) {
  const float trace = (grad[0] + grad[4] + grad[8]) / 3.0f;
  terms[0] = grad[0] - trace;
  terms[1] = grad[4] - trace;
  terms[2] = grad[8] - trace;
  terms[3] = grad[1] + grad[3];
  terms[4] = grad[2] + grad[6];
  terms[5] = grad[5] + grad[7];
}

__device__ void fill_spin_term_derivativesf(
    int degree,
    int count,
    const float* terms,
    float* derivatives) {
  const int lower_count = degree == 3 ? kSpinDeg2Count : kSpinDeg3Count;
  for (int k = 0; k < 3 * lower_count; ++k) {
    derivatives[k] = 0.0f;
  }
  const int exp3[kSpinDeg3Count][3] = {
      {3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {2, 1, 0}, {2, 0, 1},
      {1, 2, 0}, {0, 2, 1}, {1, 0, 2}, {0, 1, 2}, {1, 1, 1}};
  const int exp4[kSpinDeg4Count][3] = {
      {4, 0, 0}, {0, 4, 0}, {0, 0, 4}, {3, 1, 0}, {3, 0, 1},
      {1, 3, 0}, {0, 3, 1}, {1, 0, 3}, {0, 1, 3}, {2, 2, 0},
      {2, 0, 2}, {0, 2, 2}, {2, 1, 1}, {1, 2, 1}, {1, 1, 2}};
  for (int k = 0; k < count; ++k) {
    const int* exps = degree == 3 ? exp3[k] : exp4[k];
    for (int axis = 0; axis < 3; ++axis) {
      const int power = exps[axis];
      if (power == 0) {
        continue;
      }
      int lower[3] = {exps[0], exps[1], exps[2]};
      --lower[axis];
      const int lower_index =
          spin_monomial_index(degree - 1, lower[0], lower[1], lower[2]);
      derivatives[axis * lower_count + lower_index] +=
          static_cast<float>(power) * terms[k];
    }
  }
}

__device__ __forceinline__ void add_stf_outer_gradientf(
    const float* grad,
    const float* a,
    const float* b,
    float* grad_a,
    float* grad_b) {
  const float trace_grad = (grad[0] + grad[4] + grad[8]) / 3.0f;
  for (int p = 0; p < 3; ++p) {
    float ga = -trace_grad * b[p];
    float gb = -trace_grad * a[p];
    for (int q = 0; q < 3; ++q) {
      ga += 0.5f * (grad[3 * p + q] + grad[3 * q + p]) * b[q];
      gb += 0.5f * (grad[3 * p + q] + grad[3 * q + p]) * a[q];
    }
    grad_a[p] += ga;
    grad_b[p] += gb;
  }
}

__device__ void apply_edge_gradients_f32(
    int atom,
    int neighbor,
    int atom_stride,
    float dist,
    const float* rhat,
    float grad_weight,
    float weight_derivative,
    const float* grad_rhat,
    const float* grad_si,
    const float* grad_sj,
    double* force_soa3,
    double* mforce_soa3,
    bool accumulate_virial,
    double* virial_soa9) {
  const float grad_dist = grad_weight * weight_derivative;
  float dot_r = 0.0f;
  for (int d = 0; d < 3; ++d) {
    dot_r += grad_rhat[d] * rhat[d];
  }
  float grad_rij[3];
  for (int d = 0; d < 3; ++d) {
    grad_rij[d] =
        grad_dist * rhat[d] + (grad_rhat[d] - dot_r * rhat[d]) / dist;
    atomicAdd(force_soa3 + d * atom_stride + atom,
              static_cast<double>(grad_rij[d]));
    atomicAdd(force_soa3 + d * atom_stride + neighbor,
              -static_cast<double>(grad_rij[d]));
    atomicAdd(mforce_soa3 + d * atom_stride + atom,
              -static_cast<double>(grad_si[d]));
    atomicAdd(mforce_soa3 + d * atom_stride + neighbor,
              -static_cast<double>(grad_sj[d]));
  }
  if (accumulate_virial) {
    for (int a = 0; a < 3; ++a) {
      const double rij_a = static_cast<double>(rhat[a] * dist);
      for (int b = 0; b < 3; ++b) {
        atomicAdd(
            virial_soa9 + virial_internal_component(a * 3 + b) * atom_stride + atom,
            -rij_a * static_cast<double>(grad_rij[b]));
      }
    }
  }
}

__device__ int real_spherical_harmonics_spinf(
    const float* rhat,
    int ell,
    float* out) {
  const float x = rhat[0];
  const float y = rhat[1];
  const float z = rhat[2];
  if (ell == 2) {
    out[0] = sqrtf(static_cast<float>(15.0 / (4.0 * kPi))) * x * y;
    out[1] = sqrtf(static_cast<float>(15.0 / (4.0 * kPi))) * y * z;
    out[2] = sqrtf(static_cast<float>(5.0 / (16.0 * kPi))) *
             (2.0f * z * z - x * x - y * y);
    out[3] = sqrtf(static_cast<float>(15.0 / (4.0 * kPi))) * x * z;
    out[4] = sqrtf(static_cast<float>(15.0 / (16.0 * kPi))) *
             (x * x - y * y);
    return 5;
  }
  if (ell == 3) {
    const float rho2 = x * x + y * y;
    out[0] = sqrtf(static_cast<float>(35.0 / (32.0 * kPi))) *
             y * (3.0f * x * x - y * y);
    out[1] = sqrtf(static_cast<float>(105.0 / (4.0 * kPi))) * x * y * z;
    out[2] = sqrtf(static_cast<float>(21.0 / (32.0 * kPi))) *
             y * (4.0f * z * z - rho2);
    out[3] = sqrtf(static_cast<float>(7.0 / (16.0 * kPi))) *
             z * (2.0f * z * z - 3.0f * rho2);
    out[4] = sqrtf(static_cast<float>(21.0 / (32.0 * kPi))) *
             x * (4.0f * z * z - rho2);
    out[5] = sqrtf(static_cast<float>(105.0 / (16.0 * kPi))) *
             z * (x * x - y * y);
    out[6] = sqrtf(static_cast<float>(35.0 / (32.0 * kPi))) *
             x * (x * x - 3.0f * y * y);
    return 7;
  }
  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  out[0] = 0.75f * sqrtf(static_cast<float>(35.0 / kPi)) *
           x * y * (x2 - y2);
  out[1] = 0.75f * sqrtf(static_cast<float>(35.0 / (2.0 * kPi))) *
           y * z * (3.0f * x2 - y2);
  out[2] = 0.75f * sqrtf(static_cast<float>(5.0 / kPi)) *
           x * y * (7.0f * z2 - 1.0f);
  out[3] = 0.75f * sqrtf(static_cast<float>(5.0 / (2.0 * kPi))) *
           y * z * (7.0f * z2 - 3.0f);
  out[4] = (3.0f / 16.0f) * sqrtf(static_cast<float>(1.0 / kPi)) *
           (35.0f * z2 * z2 - 30.0f * z2 + 3.0f);
  out[5] = 0.75f * sqrtf(static_cast<float>(5.0 / (2.0 * kPi))) *
           x * z * (7.0f * z2 - 3.0f);
  out[6] = 0.375f * sqrtf(static_cast<float>(5.0 / kPi)) *
           (x2 - y2) * (7.0f * z2 - 1.0f);
  out[7] = 0.75f * sqrtf(static_cast<float>(35.0 / (2.0 * kPi))) *
           x * z * (x2 - 3.0f * y2);
  out[8] = (3.0f / 16.0f) * sqrtf(static_cast<float>(35.0 / kPi)) *
           (x2 * x2 - 6.0f * x2 * y2 + y2 * y2);
  return 9;
}

__device__ void add_real_spherical_harmonics_gradientf(
    const float* r,
    int ell,
    const float* grad_y,
    float* grad_r) {
  const float x = r[0];
  const float y = r[1];
  const float z = r[2];
  if (ell == 2) {
    const float a = sqrtf(static_cast<float>(15.0 / (4.0 * kPi)));
    const float b = sqrtf(static_cast<float>(5.0 / (16.0 * kPi)));
    const float c = sqrtf(static_cast<float>(15.0 / (16.0 * kPi)));
    grad_r[0] += grad_y[0] * a * y - grad_y[2] * 2.0f * b * x +
                 grad_y[3] * a * z + grad_y[4] * 2.0f * c * x;
    grad_r[1] += grad_y[0] * a * x + grad_y[1] * a * z -
                 grad_y[2] * 2.0f * b * y - grad_y[4] * 2.0f * c * y;
    grad_r[2] += grad_y[1] * a * y + grad_y[2] * 4.0f * b * z +
                 grad_y[3] * a * x;
    return;
  }
  if (ell == 3) {
    const float x2 = x * x;
    const float y2 = y * y;
    const float z2 = z * z;
    const float rho2 = x2 + y2;
    const float a = sqrtf(static_cast<float>(35.0 / (32.0 * kPi)));
    const float b = sqrtf(static_cast<float>(105.0 / (4.0 * kPi)));
    const float c = sqrtf(static_cast<float>(21.0 / (32.0 * kPi)));
    const float d = sqrtf(static_cast<float>(7.0 / (16.0 * kPi)));
    const float e = sqrtf(static_cast<float>(105.0 / (16.0 * kPi)));
    grad_r[0] += grad_y[0] * 6.0f * a * x * y +
                 grad_y[1] * b * y * z -
                 grad_y[2] * 2.0f * c * x * y -
                 grad_y[3] * 6.0f * d * x * z +
                 grad_y[4] * c * (4.0f * z2 - 3.0f * x2 - y2) +
                 grad_y[5] * 2.0f * e * x * z +
                 grad_y[6] * 3.0f * a * (x2 - y2);
    grad_r[1] += grad_y[0] * 3.0f * a * (x2 - y2) +
                 grad_y[1] * b * x * z +
                 grad_y[2] * c * (4.0f * z2 - x2 - 3.0f * y2) -
                 grad_y[3] * 6.0f * d * y * z -
                 grad_y[4] * 2.0f * c * x * y -
                 grad_y[5] * 2.0f * e * y * z -
                 grad_y[6] * 6.0f * a * x * y;
    grad_r[2] += grad_y[1] * b * x * y +
                 grad_y[2] * 8.0f * c * y * z +
                 grad_y[3] * d * (6.0f * z2 - 3.0f * rho2) +
                 grad_y[4] * 8.0f * c * x * z +
                 grad_y[5] * e * (x2 - y2);
    return;
  }
  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  const float a = 0.75f * sqrtf(static_cast<float>(35.0 / kPi));
  const float b = 0.75f * sqrtf(static_cast<float>(35.0 / (2.0 * kPi)));
  const float c = 0.75f * sqrtf(static_cast<float>(5.0 / kPi));
  const float d = 0.75f * sqrtf(static_cast<float>(5.0 / (2.0 * kPi)));
  const float e = (3.0f / 16.0f) * sqrtf(static_cast<float>(1.0 / kPi));
  const float f = 0.375f * sqrtf(static_cast<float>(5.0 / kPi));
  const float g = (3.0f / 16.0f) * sqrtf(static_cast<float>(35.0 / kPi));
  grad_r[0] += grad_y[0] * a * y * (3.0f * x2 - y2) +
               grad_y[1] * b * 6.0f * x * y * z +
               grad_y[2] * c * y * (7.0f * z2 - 1.0f) +
               grad_y[5] * d * z * (7.0f * z2 - 3.0f) +
               grad_y[6] * 2.0f * f * x * (7.0f * z2 - 1.0f) +
               grad_y[7] * b * z * (3.0f * x2 - 3.0f * y2) +
               grad_y[8] * g * (4.0f * x * x2 - 12.0f * x * y2);
  grad_r[1] += grad_y[0] * a * x * (x2 - 3.0f * y2) +
               grad_y[1] * b * z * (3.0f * x2 - 3.0f * y2) +
               grad_y[2] * c * x * (7.0f * z2 - 1.0f) +
               grad_y[3] * d * z * (7.0f * z2 - 3.0f) -
               grad_y[6] * 2.0f * f * y * (7.0f * z2 - 1.0f) -
               grad_y[7] * b * 6.0f * x * y * z +
               grad_y[8] * g * (-12.0f * x2 * y + 4.0f * y * y2);
  grad_r[2] += grad_y[1] * b * y * (3.0f * x2 - y2) +
               grad_y[2] * c * 14.0f * x * y * z +
               grad_y[3] * d * y * (21.0f * z2 - 3.0f) +
               grad_y[4] * e * (140.0f * z2 * z - 60.0f * z) +
               grad_y[5] * d * x * (21.0f * z2 - 3.0f) +
               grad_y[6] * f * 14.0f * z * (x2 - y2) +
               grad_y[7] * b * x * (x2 - 3.0f * y2);
}

struct SpinDensityPullSharedC4L4 {
  float rho0[12];
  float l1[36];
  float angular2[60];
  float angular3[84];
  float angular4[108];
  float geom[36];
  float rho0_dot[12];
  float l1_dot[36];
};

static_assert(
    sizeof(SpinDensityPullSharedC4L4) == 384 * sizeof(float),
    "c4/l4 fused density pulls must remain tightly packed");

template <bool AtomMajor>
struct SpinDensityForceSharedC4L4 {
  float reduce[6][32];
};

template <>
struct SpinDensityForceSharedC4L4<true> {
  SpinDensityPullSharedC4L4 pulls;
  float reduce[6][32];
  double direct_center_mforce[3];
};

template <bool AtomMajor>
__global__ void __launch_bounds__(32, 8)
accumulate_spin_density_forces_c4_l4_block_f32(
    int atom_count,
    int atom_stride,
    int struct_dim,
    float spin_cutoff,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ fp,
    const float* __restrict__ spin_edge_dx,
    const float* __restrict__ spin_edge_dy,
    const float* __restrict__ spin_edge_dz,
    const float* __restrict__ spin_edge_dist,
    const float* __restrict__ spin_edge_weights,
    const float* __restrict__ spin_edge_weight_derivatives,
    const float* __restrict__ density_rho0_cache,
    const float* __restrict__ density_l1_rdot_cache,
    const float* __restrict__ density_l1_cross_cache,
    const float* __restrict__ density_l1_stf_cache,
    const float* __restrict__ density_angular2_cache,
    const float* __restrict__ density_angular3_cache,
    const float* __restrict__ density_angular4_cache,
    const float* __restrict__ density_geom_cache,
    const float* __restrict__ density_rho0_dot_cache,
    const float* __restrict__ density_raw1_cache,
    const float* __restrict__ density_raw1_dot_cache,
    double* __restrict__ force_soa3,
    double* __restrict__ mforce_soa3) {
  constexpr int C = 4;
  constexpr int Rho0Offset = 18;
  constexpr int L1RdotOffset = 22;
  constexpr int L1CrossOffset = 26;
  constexpr int L1StfOffset = 30;
  constexpr int Angular2Offset = 34;
  constexpr int Angular3Offset = 38;
  constexpr int Angular4Offset = 42;
  constexpr int GeomOffset = 46;
  constexpr int Rho0DotOffset = 50;
  constexpr int Raw1DotOffset = 54;
  const int atom = blockIdx.x;
  const int lane = threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  __shared__ SpinDensityForceSharedC4L4<AtomMajor> shared;
  const int cache_stride = spin_component_cache_stride<AtomMajor>(atom_stride);

  const float* rho0_pull_base;
  const float* l1_pull_base;
  const float* angular2_pull_base;
  const float* angular3_pull_base;
  const float* angular4_pull_base;
  const float* geom_pull_base;
  const float* rho0_dot_pull_base;
  const float* l1_dot_pull_base;
  if constexpr (AtomMajor) {
    constexpr unsigned int FullWarpMask = 0xffffffffu;
    const double spin_lane = lane < 3
        ? spins_soa3[lane * atom_stride + atom]
        : 0.0;
    const double si0[3] = {
        __shfl_sync(FullWarpMask, spin_lane, 0),
        __shfl_sync(FullWarpMask, spin_lane, 1),
        __shfl_sync(FullWarpMask, spin_lane, 2)};
    const double spin_trace =
        (si0[0] * si0[0] + si0[1] * si0[1] + si0[2] * si0[2]) / 3.0;

    for (int component = lane; component < C * 3; component += blockDim.x) {
      const int c = component / 3;
      const double alpha0 = static_cast<double>(
          fp[atom + atom_stride * (struct_dim + Rho0Offset + c)]);
      const double alpha0_dot = static_cast<double>(
          fp[atom + atom_stride * (struct_dim + Rho0DotOffset + c)]);
      const int index = spin_component_cache_index<true, C * 3>(
          atom_stride, atom, component);
      const double rho0 = static_cast<double>(density_rho0_cache[index]);
      const double rho0_dot =
          static_cast<double>(density_rho0_dot_cache[index]);
      shared.pulls.rho0[component] =
          static_cast<float>(2.0 * alpha0 * rho0 + alpha0_dot * rho0_dot);
      shared.pulls.rho0_dot[component] =
          static_cast<float>(alpha0_dot * rho0);
    }

    for (int component = lane; component < C * 9; component += blockDim.x) {
      const int c = component / 9;
      const int k = component - c * 9;
      const double alpha_rdot = static_cast<double>(
          fp[atom + atom_stride * (struct_dim + L1RdotOffset + c)]);
      const double alpha_cross = static_cast<double>(
          fp[atom + atom_stride * (struct_dim + L1CrossOffset + c)]);
      const double alpha_stf = static_cast<double>(
          fp[atom + atom_stride * (struct_dim + L1StfOffset + c)]);
      const double alpha_raw1 = static_cast<double>(
          fp[atom + atom_stride * (struct_dim + Raw1DotOffset + c)]);
      const double rdot = static_cast<double>(
          density_l1_rdot_cache[
              spin_component_cache_index<true, C>(atom_stride, atom, c)]);
      const double cross[3] = {
          static_cast<double>(
              density_l1_cross_cache[
                  spin_component_cache_index<true, C * 3>(
                      atom_stride, atom, c * 3)]),
          static_cast<double>(
              density_l1_cross_cache[
                  spin_component_cache_index<true, C * 3>(
                      atom_stride, atom, c * 3 + 1)]),
          static_cast<double>(
              density_l1_cross_cache[
                  spin_component_cache_index<true, C * 3>(
                      atom_stride, atom, c * 3 + 2)])};
      const double g_cross[3] = {
          2.0 * alpha_cross * cross[0],
          2.0 * alpha_cross * cross[1],
          2.0 * alpha_cross * cross[2]};
      double mat = 0.0;
      if (k == 0 || k == 4 || k == 8) {
        mat += 2.0 * alpha_rdot * rdot;
      } else if (k == 1) {
        mat += g_cross[2];
      } else if (k == 2) {
        mat -= g_cross[1];
      } else if (k == 3) {
        mat -= g_cross[2];
      } else if (k == 5) {
        mat += g_cross[0];
      } else if (k == 6) {
        mat += g_cross[1];
      } else if (k == 7) {
        mat -= g_cross[0];
      }
      const int index = spin_component_cache_index<true, C * 9>(
          atom_stride, atom, component);
      const double stf = static_cast<double>(density_l1_stf_cache[index]);
      const double raw = static_cast<double>(density_raw1_cache[index]);
      const double raw_dot =
          static_cast<double>(density_raw1_dot_cache[index]);
      shared.pulls.l1[component] = static_cast<float>(
          mat + 2.0 * alpha_stf * stf + alpha_raw1 * raw_dot);
      shared.pulls.l1_dot[component] =
          static_cast<float>(alpha_raw1 * raw);

      const double alpha_geom = static_cast<double>(
          fp[atom + atom_stride * (struct_dim + GeomOffset + c)]);
      const int a = k / 3;
      const int b = k % 3;
      double ss = 0.5 * (si0[a] * si0[b] + si0[b] * si0[a]);
      if (a == b) {
        ss -= spin_trace;
      }
      shared.pulls.geom[component] =
          static_cast<float>(alpha_geom * ss);
    }

    for (int component = lane; component < C * 15; component += blockDim.x) {
      const int c = component / 15;
      const double alpha = static_cast<double>(
          fp[atom + atom_stride * (struct_dim + Angular2Offset + c)]);
      const int index = spin_component_cache_index<true, C * 15>(
          atom_stride, atom, component);
      shared.pulls.angular2[component] = static_cast<float>(
          2.0 * alpha * static_cast<double>(density_angular2_cache[index]));
    }
    for (int component = lane; component < C * 21; component += blockDim.x) {
      const int c = component / 21;
      const double alpha = static_cast<double>(
          fp[atom + atom_stride * (struct_dim + Angular3Offset + c)]);
      const int index = spin_component_cache_index<true, C * 21>(
          atom_stride, atom, component);
      shared.pulls.angular3[component] = static_cast<float>(
          2.0 * alpha * static_cast<double>(density_angular3_cache[index]));
    }
    for (int component = lane; component < C * 27; component += blockDim.x) {
      const int c = component / 27;
      const double alpha = static_cast<double>(
          fp[atom + atom_stride * (struct_dim + Angular4Offset + c)]);
      const int index = spin_component_cache_index<true, C * 27>(
          atom_stride, atom, component);
      shared.pulls.angular4[component] = static_cast<float>(
          2.0 * alpha * static_cast<double>(density_angular4_cache[index]));
    }

    if (lane < 3) {
      double grad_spin_i_direct = 0.0;
      for (int c = 0; c < C; ++c) {
        const double alpha_geom = static_cast<double>(
            fp[atom + atom_stride * (struct_dim + GeomOffset + c)]);
        double gs = 0.0;
        for (int b = 0; b < 3; ++b) {
          const double geom_ab = static_cast<double>(
              density_geom_cache[
                  spin_component_cache_index<true, C * 9>(
                      atom_stride, atom, c * 9 + 3 * lane + b)]);
          const double geom_ba = static_cast<double>(
              density_geom_cache[
                  spin_component_cache_index<true, C * 9>(
                      atom_stride, atom, c * 9 + 3 * b + lane)]);
          gs += (geom_ab + geom_ba) * si0[b];
        }
        grad_spin_i_direct += alpha_geom * gs;
      }
      shared.direct_center_mforce[lane] = -grad_spin_i_direct;
    }
    __syncthreads();

    rho0_pull_base = shared.pulls.rho0;
    l1_pull_base = shared.pulls.l1;
    angular2_pull_base = shared.pulls.angular2;
    angular3_pull_base = shared.pulls.angular3;
    angular4_pull_base = shared.pulls.angular4;
    geom_pull_base = shared.pulls.geom;
    rho0_dot_pull_base = shared.pulls.rho0_dot;
    l1_dot_pull_base = shared.pulls.l1_dot;
  } else {
    rho0_pull_base = density_rho0_cache +
        spin_component_cache_index<false, C * 3>(atom_stride, atom, 0);
    l1_pull_base = density_l1_stf_cache +
        spin_component_cache_index<false, C * 9>(atom_stride, atom, 0);
    angular2_pull_base = density_angular2_cache +
        spin_component_cache_index<false, C * 15>(atom_stride, atom, 0);
    angular3_pull_base = density_angular3_cache +
        spin_component_cache_index<false, C * 21>(atom_stride, atom, 0);
    angular4_pull_base = density_angular4_cache +
        spin_component_cache_index<false, C * 27>(atom_stride, atom, 0);
    geom_pull_base = density_raw1_cache +
        spin_component_cache_index<false, C * 9>(atom_stride, atom, 0);
    rho0_dot_pull_base = density_rho0_dot_cache +
        spin_component_cache_index<false, C * 3>(atom_stride, atom, 0);
    l1_dot_pull_base = density_raw1_dot_cache +
        spin_component_cache_index<false, C * 9>(atom_stride, atom, 0);
  }

  float center_force[3] = {};
  float center_mforce[3] = {};
  const int radial_count = nn_radial[atom];
  for (int slot = lane; slot < radial_count; slot += blockDim.x) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    float rhat[3];
    float dist = 0.0f;
    float si[3];
    float sj[3];
    load_spin_edge_cached_f32(
        atom,
        neighbor,
        atom_stride,
        slot,
        spins_soa3,
        spin_edge_dx,
        spin_edge_dy,
        spin_edge_dz,
        spin_edge_dist,
        rhat,
        dist,
        si,
        sj);
    if (dist <= 1.0e-12f || dist >= spin_cutoff) {
      continue;
    }

    float weights[C];
    float weight_derivatives[C];
    load_spin_edge_weight_derivative_cache_f32(
        atom_stride,
        slot,
        atom,
        spin_edge_weights,
        spin_edge_weight_derivatives,
        weights,
        weight_derivatives);

    const float dot = dot3f(si, sj);
    float grad_weight[C] = {};
    float grad_rhat[3] = {};
    float grad_si[3] = {};
    float grad_sj[3] = {};
    float grad_dot = 0.0f;

    const float sj2 = dot3f(sj, sj);
    const float ri_dot_si = dot3f(rhat, si);
    const float ri_dot_sj = dot3f(rhat, sj);
    const float bond_axis = ri_dot_si * ri_dot_sj;
    float scalar_weighted = 0.0f;
    for (int c = 0; c < C; ++c) {
      const float alpha = fp[atom + atom_stride * (struct_dim + 2 + c)];
      grad_weight[c] += alpha * dot;
      scalar_weighted += alpha * weights[c];
    }
    grad_dot += scalar_weighted;
    scalar_weighted = 0.0f;
    for (int c = 0; c < C; ++c) {
      const float alpha = fp[atom + atom_stride * (struct_dim + 2 + C + c)];
      grad_weight[c] += alpha * dot * dot;
      scalar_weighted += alpha * weights[c];
    }
    grad_dot += 2.0f * dot * scalar_weighted;
    scalar_weighted = 0.0f;
    for (int c = 0; c < C; ++c) {
      const float alpha = fp[atom + atom_stride * (struct_dim + 2 + 2 * C + c)];
      grad_weight[c] += alpha * sj2;
      scalar_weighted += alpha * weights[c];
    }
    for (int d = 0; d < 3; ++d) {
      grad_sj[d] += 2.0f * scalar_weighted * sj[d];
    }
    scalar_weighted = 0.0f;
    for (int c = 0; c < C; ++c) {
      const float alpha = fp[atom + atom_stride * (struct_dim + 2 + 3 * C + c)];
      grad_weight[c] += alpha * bond_axis;
      scalar_weighted += alpha * weights[c];
    }
    for (int d = 0; d < 3; ++d) {
      grad_si[d] += scalar_weighted * ri_dot_sj * rhat[d];
      grad_sj[d] += scalar_weighted * ri_dot_si * rhat[d];
      grad_rhat[d] +=
          scalar_weighted * (ri_dot_sj * si[d] + ri_dot_si * sj[d]);
    }

    for (int c = 0; c < C; ++c) {
      const float b[3] = {
          rho0_pull_base[(c * 3) * cache_stride],
          rho0_pull_base[(c * 3 + 1) * cache_stride],
          rho0_pull_base[(c * 3 + 2) * cache_stride]};
      const float bd[3] = {
          rho0_dot_pull_base[(c * 3) * cache_stride],
          rho0_dot_pull_base[(c * 3 + 1) * cache_stride],
          rho0_dot_pull_base[(c * 3 + 2) * cache_stride]};
      const float u[3] = {
          b[0] + dot * bd[0],
          b[1] + dot * bd[1],
          b[2] + dot * bd[2]};
      const float w = weights[c];
      grad_weight[c] += dot3f(sj, u);
      for (int d = 0; d < 3; ++d) {
        grad_sj[d] += w * u[d];
      }
      grad_dot += w * dot3f(sj, bd);

      const float* gbase = geom_pull_base + c * 9 * cache_stride;
      const float* mbase = l1_pull_base + c * 9 * cache_stride;
      const float* mdbase = l1_dot_pull_base + c * 9 * cache_stride;
      const float kr[3] = {
          gbase[0 * cache_stride] * rhat[0] +
              gbase[1 * cache_stride] * rhat[1] +
              gbase[2 * cache_stride] * rhat[2],
          gbase[3 * cache_stride] * rhat[0] +
              gbase[4 * cache_stride] * rhat[1] +
              gbase[5 * cache_stride] * rhat[2],
          gbase[6 * cache_stride] * rhat[0] +
              gbase[7 * cache_stride] * rhat[1] +
              gbase[8 * cache_stride] * rhat[2]};
      const float v1[3] = {
          mbase[0 * cache_stride] * sj[0] + mbase[1 * cache_stride] * sj[1] +
              mbase[2 * cache_stride] * sj[2],
          mbase[3 * cache_stride] * sj[0] + mbase[4 * cache_stride] * sj[1] +
              mbase[5 * cache_stride] * sj[2],
          mbase[6 * cache_stride] * sj[0] + mbase[7 * cache_stride] * sj[1] +
              mbase[8 * cache_stride] * sj[2]};
      const float v2[3] = {
          mdbase[0 * cache_stride] * sj[0] +
              mdbase[1 * cache_stride] * sj[1] +
              mdbase[2 * cache_stride] * sj[2],
          mdbase[3 * cache_stride] * sj[0] +
              mdbase[4 * cache_stride] * sj[1] +
              mdbase[5 * cache_stride] * sj[2],
          mdbase[6 * cache_stride] * sj[0] +
              mdbase[7 * cache_stride] * sj[1] +
              mdbase[8 * cache_stride] * sj[2]};
      const float u1[3] = {
          v1[0] + dot * v2[0],
          v1[1] + dot * v2[1],
          v1[2] + dot * v2[2]};
      const float mt[3] = {
          mbase[0 * cache_stride] * rhat[0] +
              mbase[3 * cache_stride] * rhat[1] +
              mbase[6 * cache_stride] * rhat[2] +
              dot * (mdbase[0 * cache_stride] * rhat[0] +
                     mdbase[3 * cache_stride] * rhat[1] +
                     mdbase[6 * cache_stride] * rhat[2]),
          mbase[1 * cache_stride] * rhat[0] +
              mbase[4 * cache_stride] * rhat[1] +
              mbase[7 * cache_stride] * rhat[2] +
              dot * (mdbase[1 * cache_stride] * rhat[0] +
                     mdbase[4 * cache_stride] * rhat[1] +
                     mdbase[7 * cache_stride] * rhat[2]),
          mbase[2 * cache_stride] * rhat[0] +
              mbase[5 * cache_stride] * rhat[1] +
              mbase[8 * cache_stride] * rhat[2] +
              dot * (mdbase[2 * cache_stride] * rhat[0] +
                     mdbase[5 * cache_stride] * rhat[1] +
                     mdbase[8 * cache_stride] * rhat[2])};
      for (int d = 0; d < 3; ++d) {
        grad_weight[c] += rhat[d] * (u1[d] + kr[d]);
        grad_rhat[d] += w * (u1[d] + 2.0f * kr[d]);
        grad_sj[d] += w * mt[d];
      }
      grad_dot += w * dot3f(rhat, v2);
    }

    for (int ell = 2; ell <= 4; ++ell) {
      const int width = (2 * ell + 1) * 3;
      const float* angular =
          ell == 2 ? angular2_pull_base :
          ell == 3 ? angular3_pull_base : angular4_pull_base;
      float ylm[9];
      const int ylm_width = real_spherical_harmonics_spinf(rhat, ell, ylm);
      float ge[27] = {};
      for (int c = 0; c < C; ++c) {
        for (int m = 0; m < ylm_width; ++m) {
          for (int d = 0; d < 3; ++d) {
            const int k = m * 3 + d;
            const float gd = angular[(c * width + k) * cache_stride];
            grad_weight[c] += gd * ylm[m] * sj[d];
            ge[k] += gd * weights[c];
          }
        }
      }
      float grad_ylm[9] = {};
      for (int m = 0; m < ylm_width; ++m) {
        for (int d = 0; d < 3; ++d) {
          const float g = ge[m * 3 + d];
          grad_sj[d] += g * ylm[m];
          grad_ylm[m] += g * sj[d];
        }
      }
      add_real_spherical_harmonics_gradientf(rhat, ell, grad_ylm, grad_rhat);
    }

    for (int d = 0; d < 3; ++d) {
      grad_si[d] += grad_dot * sj[d];
      grad_sj[d] += grad_dot * si[d];
    }

    float grad_dist = 0.0f;
    for (int c = 0; c < C; ++c) {
      grad_dist += grad_weight[c] * weight_derivatives[c];
    }
    float dot_r = 0.0f;
    for (int d = 0; d < 3; ++d) {
      dot_r += grad_rhat[d] * rhat[d];
    }
    for (int d = 0; d < 3; ++d) {
      const float grad_rij =
          grad_dist * rhat[d] + (grad_rhat[d] - dot_r * rhat[d]) / dist;
      center_force[d] += grad_rij;
      center_mforce[d] -= grad_si[d];
      atomicAdd(force_soa3 + d * atom_stride + neighbor,
                -static_cast<double>(grad_rij));
      atomicAdd(mforce_soa3 + d * atom_stride + neighbor,
                -static_cast<double>(grad_sj[d]));
    }
  }

  float (*reduce)[32] = shared.reduce;
  for (int d = 0; d < 3; ++d) {
    reduce[d][lane] = center_force[d];
    reduce[d + 3][lane] = center_mforce[d];
  }
  __syncthreads();
  for (int stride = 16; stride > 0; stride >>= 1) {
    if (lane < stride) {
      for (int d = 0; d < 6; ++d) {
        reduce[d][lane] += reduce[d][lane + stride];
      }
    }
    __syncthreads();
  }
  if (lane == 0) {
    for (int d = 0; d < 3; ++d) {
      atomicAdd(force_soa3 + d * atom_stride + atom,
                static_cast<double>(reduce[d][0]));
      double center_mforce_total = static_cast<double>(reduce[d + 3][0]);
      if constexpr (AtomMajor) {
        center_mforce_total += shared.direct_center_mforce[d];
      }
      atomicAdd(mforce_soa3 + d * atom_stride + atom,
                center_mforce_total);
    }
  }
}

template <bool AtomMajor>
__global__ void build_spin_chiral_finalize_c4_l4_f32(
    int atom_count,
    int atom_stride,
    int struct_dim,
    float spin_cutoff,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ spin_edge_dx,
    const float* __restrict__ spin_edge_dy,
    const float* __restrict__ spin_edge_dz,
    const float* __restrict__ spin_edge_dist,
    const float* __restrict__ spin_edge_weights,
    const float* __restrict__ density_geom_cache,
    const float* __restrict__ chiral_polar_cache,
    const float* __restrict__ chiral_octupoles_raw_cache,
    const float* __restrict__ chiral_hexadecapoles_raw_cache,
    float* __restrict__ chiral_chirals_cache,
    float* __restrict__ chiral_pseudodevs_cache,
    float* __restrict__ descriptors) {
  constexpr int C = 4;
  constexpr int ChiC = 2;
  constexpr int ChiralOffset = 58;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int atom = tid / C;
  const int c = tid - atom * C;
  if (atom >= atom_count) {
    return;
  }

  float geom[9];
  float polar[3];
  for (int k = 0; k < 9; ++k) {
    geom[k] = density_geom_cache[
        spin_component_cache_index<AtomMajor, C * 9>(
            atom_stride, atom, c * 9 + k)];
  }
  for (int k = 0; k < 3; ++k) {
    polar[k] = chiral_polar_cache[
        spin_component_cache_index<AtomMajor, C * 3>(
            atom_stride, atom, c * 3 + k)];
  }

  float chiral_value = 0.0f;
  if (c < ChiC) {
    float octupoles_raw[kSpinDeg3Count];
    float hexadecapoles_raw[kSpinDeg4Count];
    for (int k = 0; k < kSpinDeg3Count; ++k) {
      octupoles_raw[k] =
          chiral_octupoles_raw_cache[
              spin_component_cache_index<AtomMajor, ChiC * kSpinDeg3Count>(
                  atom_stride, atom, c * kSpinDeg3Count + k)];
    }
    for (int k = 0; k < kSpinDeg4Count; ++k) {
      hexadecapoles_raw[k] =
          chiral_hexadecapoles_raw_cache[
              spin_component_cache_index<AtomMajor, ChiC * kSpinDeg4Count>(
                  atom_stride, atom, c * kSpinDeg4Count + k)];
    }
    for (int term = 0; term < kSpinChiralQohSymCount; ++term) {
      const unsigned short packed = kSpinChiralQohSymPacked[term];
      const int q = packed >> 8;
      const int o = (packed >> 4) & 0x0f;
      const int h = packed & 0x0f;
      chiral_value += kSpinChiralQohSymCoeff[term] *
                      geom[q] * octupoles_raw[o] * hexadecapoles_raw[h];
    }
    chiral_chirals_cache[
        spin_component_cache_index<AtomMajor, ChiC>(atom_stride, atom, c)] =
        chiral_value;
  }

  float pseudodevs[9] = {};
  const int radial_count = nn_radial[atom];
  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    float rhat[3];
    float dist = 0.0f;
    float si[3];
    float sj[3];
    load_spin_edge_cached_f32(
        atom,
        neighbor,
        atom_stride,
        slot,
        spins_soa3,
        spin_edge_dx,
        spin_edge_dy,
        spin_edge_dz,
        spin_edge_dist,
        rhat,
        dist,
        si,
        sj);
    if (dist <= 1.0e-12f || dist >= spin_cutoff) {
      continue;
    }
    const float weight =
        spin_edge_weights[spin_edge_cache_index(atom_stride, slot, atom, c)];
    float qu[3] = {0.0f, 0.0f, 0.0f};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        qu[a] += geom[3 * a + b] * rhat[b];
      }
    }
    float axis[3];
    cross3f(rhat, qu, axis);
    float pseudo[9];
    stf_outer3f(axis, rhat, pseudo);
    for (int k = 0; k < 9; ++k) {
      pseudodevs[k] += weight * pseudo[k];
    }
  }
  for (int k = 0; k < 9; ++k) {
    chiral_pseudodevs_cache[
        spin_component_cache_index<AtomMajor, C * 9>(
            atom_stride, atom, c * 9 + k)] = pseudodevs[k];
  }

  float chiral_q0 = 0.0f;
  float chiral_q1 = 0.0f;
  float chiral_q2 = 0.0f;
  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    float rhat[3];
    float dist = 0.0f;
    float si[3];
    float sj[3];
    load_spin_edge_cached_f32(
        atom,
        neighbor,
        atom_stride,
        slot,
        spins_soa3,
        spin_edge_dx,
        spin_edge_dy,
        spin_edge_dz,
        spin_edge_dist,
        rhat,
        dist,
        si,
        sj);
    if (dist <= 1.0e-12f || dist >= spin_cutoff) {
      continue;
    }
    const float weight =
        spin_edge_weights[spin_edge_cache_index(atom_stride, slot, atom, c)];
    float spin_cross[3];
    cross3f(si, sj, spin_cross);
    if (c < ChiC) {
      chiral_q0 += weight * dot3f(spin_cross, rhat) * chiral_value;
    }
    float axis[3];
    cross3f(polar, rhat, axis);
    chiral_q1 += weight * dot3f(spin_cross, axis);
    float pseudo_axis[3] = {0.0f, 0.0f, 0.0f};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        pseudo_axis[a] += pseudodevs[3 * a + b] * rhat[b];
      }
    }
    chiral_q2 += weight * dot3f(spin_cross, pseudo_axis);
  }
  if (c < ChiC) {
    descriptors[atom + atom_stride * (struct_dim + ChiralOffset + c)] =
        chiral_q0;
  }
  descriptors[atom + atom_stride * (struct_dim + ChiralOffset + ChiC + c)] =
      chiral_q1;
  descriptors[atom + atom_stride * (struct_dim + ChiralOffset + ChiC + C + c)] =
      chiral_q2;
}

template <bool AtomMajor>
__global__ void __launch_bounds__(32, 8)
accumulate_spin_chiral_forces_c4_l4_cached_f32(
    int atom_count,
    int atom_stride,
    int struct_dim,
    float spin_cutoff,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ fp,
    const float* __restrict__ spin_edge_dx,
    const float* __restrict__ spin_edge_dy,
    const float* __restrict__ spin_edge_dz,
    const float* __restrict__ spin_edge_dist,
    const float* __restrict__ spin_edge_weights,
    const float* __restrict__ spin_edge_weight_derivatives,
    const float* __restrict__ density_geom_cache,
    const float* __restrict__ chiral_polar_cache,
    const float* __restrict__ chiral_octupoles_raw_cache,
    const float* __restrict__ chiral_hexadecapoles_raw_cache,
    const float* __restrict__ chiral_chirals_cache,
    const float* __restrict__ chiral_pseudodevs_cache,
    double* __restrict__ force_soa3,
    double* __restrict__ mforce_soa3,
    bool accumulate_virial,
    double* __restrict__ virial_soa9) {
  constexpr int C = 4;
  constexpr int ChiC = 2;
  constexpr int BaseOffset = 58;
  constexpr int PolarOffset = BaseOffset + ChiC;
  constexpr int PseudoOffset = PolarOffset + C;
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const int radial_count = nn_radial[atom];
  float alpha_chiral_pull[ChiC] = {};
  float alpha_polar_pull[C] = {};
  float alpha_pseudo_pull[C] = {};
  for (int c = 0; c < ChiC; ++c) {
    alpha_chiral_pull[c] =
        fp[atom + atom_stride * (struct_dim + BaseOffset + c)];
  }
  for (int c = 0; c < C; ++c) {
    alpha_polar_pull[c] =
        fp[atom + atom_stride * (struct_dim + PolarOffset + c)];
    alpha_pseudo_pull[c] =
        fp[atom + atom_stride * (struct_dim + PseudoOffset + c)];
  }

  float geom[C * 9] = {};
  float polar[C * 3] = {};
  float octupoles_raw[ChiC * kSpinDeg3Count] = {};
  float hexadecapoles_raw[ChiC * kSpinDeg4Count] = {};
  float chirals[ChiC] = {};
  float pseudodevs[C * 9] = {};
  for (int c = 0; c < C; ++c) {
    for (int k = 0; k < 3; ++k) {
      polar[c * 3 + k] = chiral_polar_cache[
          spin_component_cache_index<AtomMajor, C * 3>(
              atom_stride, atom, c * 3 + k)];
    }
    for (int k = 0; k < 9; ++k) {
      geom[c * 9 + k] = density_geom_cache[
          spin_component_cache_index<AtomMajor, C * 9>(
              atom_stride, atom, c * 9 + k)];
      pseudodevs[c * 9 + k] =
          chiral_pseudodevs_cache[
              spin_component_cache_index<AtomMajor, C * 9>(
                  atom_stride, atom, c * 9 + k)];
    }
  }
  for (int c = 0; c < ChiC; ++c) {
    for (int k = 0; k < kSpinDeg3Count; ++k) {
      octupoles_raw[c * kSpinDeg3Count + k] =
          chiral_octupoles_raw_cache[
              spin_component_cache_index<AtomMajor, ChiC * kSpinDeg3Count>(
                  atom_stride, atom, c * kSpinDeg3Count + k)];
    }
    for (int k = 0; k < kSpinDeg4Count; ++k) {
      hexadecapoles_raw[c * kSpinDeg4Count + k] =
          chiral_hexadecapoles_raw_cache[
              spin_component_cache_index<AtomMajor, ChiC * kSpinDeg4Count>(
                  atom_stride, atom, c * kSpinDeg4Count + k)];
    }
    chirals[c] = chiral_chirals_cache[
        spin_component_cache_index<AtomMajor, ChiC>(atom_stride, atom, c)];
  }

  float grad_chi[ChiC] = {};
  float grad_polar[C * 3] = {};
  float grad_pseudodev[C * 9] = {};
  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    float rhat[3];
    float dist = 0.0f;
    float si[3];
    float sj[3];
    load_spin_edge_cached_f32(
        atom,
        neighbor,
        atom_stride,
        slot,
        spins_soa3,
        spin_edge_dx,
        spin_edge_dy,
        spin_edge_dz,
        spin_edge_dist,
        rhat,
        dist,
        si,
        sj);
    if (dist <= 1.0e-12f || dist >= spin_cutoff) {
      continue;
    }
    float weights[C] = {};
    load_spin_edge_weight_cache_f32(
        atom_stride, slot, atom, spin_edge_weights, weights);
    float x[3];
    cross3f(si, sj, x);
    const float xu = dot3f(x, rhat);
    for (int c = 0; c < ChiC; ++c) {
      grad_chi[c] += alpha_chiral_pull[c] * weights[c] * xu;
    }
    for (int c = 0; c < C; ++c) {
      float axis[3];
      cross3f(polar + c * 3, rhat, axis);
      const float alpha_polar = alpha_polar_pull[c];
      float gaxis_polar[3] = {
          alpha_polar * weights[c] * x[0],
          alpha_polar * weights[c] * x[1],
          alpha_polar * weights[c] * x[2]};
      float gp[3];
      cross3f(rhat, gaxis_polar, gp);
      for (int d = 0; d < 3; ++d) {
        grad_polar[c * 3 + d] += gp[d];
      }
      const float alpha_pseudo = alpha_pseudo_pull[c];
      float gaxis_pseudo[3] = {
          alpha_pseudo * weights[c] * x[0],
          alpha_pseudo * weights[c] * x[1],
          alpha_pseudo * weights[c] * x[2]};
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          grad_pseudodev[c * 9 + 3 * a + b] +=
              gaxis_pseudo[a] * rhat[b];
        }
      }
    }
  }

  float grad_Q[C * 9] = {};
  float grad_O_terms[ChiC * kSpinDeg3Count] = {};
  float grad_H_terms[ChiC * kSpinDeg4Count] = {};
  for (int c = 0; c < ChiC; ++c) {
    const float g = grad_chi[c];
    const float* qmat = geom + c * 9;
    const float* oct_raw = octupoles_raw + c * kSpinDeg3Count;
    const float* hex_raw = hexadecapoles_raw + c * kSpinDeg4Count;
    float* gQ = grad_Q + c * 9;
    float* gO_terms = grad_O_terms + c * kSpinDeg3Count;
    float* gH_terms = grad_H_terms + c * kSpinDeg4Count;
    for (int term = 0; term < kSpinChiralQohSymCount; ++term) {
      const unsigned short packed = kSpinChiralQohSymPacked[term];
      const int q = packed >> 8;
      const int o = (packed >> 4) & 0x0f;
      const int h = packed & 0x0f;
      const float scale =
          g * kSpinChiralQohSymCoeff[term];
      const float o_value = oct_raw[o];
      const float h_value = hex_raw[h];
      gQ[q] += scale * o_value * h_value;
      const float qscale = scale * qmat[q];
      gO_terms[o] += qscale * h_value;
      gH_terms[h] += qscale * o_value;
    }
  }

  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    float rhat[3];
    float dist = 0.0f;
    float si[3];
    float sj[3];
    load_spin_edge_cached_f32(
        atom,
        neighbor,
        atom_stride,
        slot,
        spins_soa3,
        spin_edge_dx,
        spin_edge_dy,
        spin_edge_dz,
        spin_edge_dist,
        rhat,
        dist,
        si,
        sj);
    if (dist <= 1.0e-12f || dist >= spin_cutoff) {
      continue;
    }
    float weights[C] = {};
    load_spin_edge_weight_cache_f32(
        atom_stride, slot, atom, spin_edge_weights, weights);
    for (int c = 0; c < C; ++c) {
      const float* gpd = grad_pseudodev + c * 9;
      const float* qmat = geom + c * 9;
      float qu[3] = {
          qmat[0] * rhat[0] + qmat[1] * rhat[1] + qmat[2] * rhat[2],
          qmat[3] * rhat[0] + qmat[4] * rhat[1] + qmat[5] * rhat[2],
          qmat[6] * rhat[0] + qmat[7] * rhat[1] + qmat[8] * rhat[2]};
      float pseudo_axis[3];
      cross3f(rhat, qu, pseudo_axis);
      float gpseudo[9];
      for (int k = 0; k < 9; ++k) {
        gpseudo[k] = gpd[k] * weights[c];
      }
      float g_axis[3] = {0.0f, 0.0f, 0.0f};
      float gu2[3] = {0.0f, 0.0f, 0.0f};
      add_stf_outer_gradientf(gpseudo, pseudo_axis, rhat, g_axis, gu2);
      float g_qu[3];
      cross3f(g_axis, rhat, g_qu);
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          grad_Q[c * 9 + 3 * a + b] += g_qu[a] * rhat[b];
        }
      }
    }
  }

  float grad_Q_terms[C * kSpinDeg2Count] = {};
  float grad_O_derivatives[ChiC * 3 * kSpinDeg2Count] = {};
  float grad_H_derivatives[ChiC * 3 * kSpinDeg3Count] = {};
  for (int c = 0; c < C; ++c) {
    project_rank2_spin_gradientf(
        grad_Q + c * 9, grad_Q_terms + c * kSpinDeg2Count);
  }
  for (int c = 0; c < ChiC; ++c) {
    fill_spin_term_derivativesf(
        3,
        kSpinDeg3Count,
        grad_O_terms + c * kSpinDeg3Count,
        grad_O_derivatives + c * 3 * kSpinDeg2Count);
    fill_spin_term_derivativesf(
        4,
        kSpinDeg4Count,
        grad_H_terms + c * kSpinDeg4Count,
        grad_H_derivatives + c * 3 * kSpinDeg3Count);
  }

  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    float rhat[3];
    float dist = 0.0f;
    float si[3];
    float sj[3];
    load_spin_edge_cached_f32(
        atom,
        neighbor,
        atom_stride,
        slot,
        spins_soa3,
        spin_edge_dx,
        spin_edge_dy,
        spin_edge_dz,
        spin_edge_dist,
        rhat,
        dist,
        si,
        sj);
    if (dist <= 1.0e-12f || dist >= spin_cutoff) {
      continue;
    }
    float weights[C] = {};
    float weight_derivatives[C] = {};
    load_spin_edge_weight_derivative_cache_f32(
        atom_stride,
        slot,
        atom,
        spin_edge_weights,
        spin_edge_weight_derivatives,
        weights,
        weight_derivatives);

    float x[3];
    cross3f(si, sj, x);
    float grad_weight[C] = {};
    float grad_rhat[3] = {0.0f, 0.0f, 0.0f};
    float grad_si[3] = {0.0f, 0.0f, 0.0f};
    float grad_sj[3] = {0.0f, 0.0f, 0.0f};
    float gx[3] = {0.0f, 0.0f, 0.0f};
    for (int c = 0; c < ChiC; ++c) {
      const float alpha = alpha_chiral_pull[c];
      const float xu = dot3f(x, rhat);
      grad_weight[c] += alpha * xu * chirals[c];
      for (int d = 0; d < 3; ++d) {
        gx[d] += alpha * weights[c] * chirals[c] * rhat[d];
        grad_rhat[d] += alpha * weights[c] * chirals[c] * x[d];
      }
    }
    for (int c = 0; c < C; ++c) {
      float axis[3];
      cross3f(polar + c * 3, rhat, axis);
      const float alpha_polar = alpha_polar_pull[c];
      const float xa = dot3f(x, axis);
      grad_weight[c] += alpha_polar * xa;
      float gaxis_polar[3] = {
          alpha_polar * weights[c] * x[0],
          alpha_polar * weights[c] * x[1],
          alpha_polar * weights[c] * x[2]};
      float gu_part[3];
      cross3f(gaxis_polar, polar + c * 3, gu_part);
      for (int d = 0; d < 3; ++d) {
        gx[d] += alpha_polar * weights[c] * axis[d];
        grad_rhat[d] += gu_part[d];
      }

      const float alpha_pseudo = alpha_pseudo_pull[c];
      const float* pdev = pseudodevs + c * 9;
      float pseudo_axis[3] = {
          pdev[0] * rhat[0] + pdev[1] * rhat[1] + pdev[2] * rhat[2],
          pdev[3] * rhat[0] + pdev[4] * rhat[1] + pdev[5] * rhat[2],
          pdev[6] * rhat[0] + pdev[7] * rhat[1] + pdev[8] * rhat[2]};
      const float xpa = dot3f(x, pseudo_axis);
      grad_weight[c] += alpha_pseudo * xpa;
      float gaxis_pseudo[3] = {
          alpha_pseudo * weights[c] * x[0],
          alpha_pseudo * weights[c] * x[1],
          alpha_pseudo * weights[c] * x[2]};
      for (int d = 0; d < 3; ++d) {
        gx[d] += alpha_pseudo * weights[c] * pseudo_axis[d];
      }
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          grad_rhat[b] += pdev[3 * a + b] * gaxis_pseudo[a];
        }
      }
    }
    float gsi_part[3];
    float gsj_part[3];
    cross3f(sj, gx, gsi_part);
    cross3f(gx, si, gsj_part);
    for (int d = 0; d < 3; ++d) {
      grad_si[d] += gsi_part[d];
      grad_sj[d] += gsj_part[d];
    }

    for (int c = 0; c < C; ++c) {
      const float* gp = grad_polar + c * 3;
      grad_weight[c] += dot3f(gp, rhat);
      for (int d = 0; d < 3; ++d) {
        grad_rhat[d] += weights[c] * gp[d];
      }

      const float* gpd = grad_pseudodev + c * 9;
      const float* qmat = geom + c * 9;
      float qu[3] = {
          qmat[0] * rhat[0] + qmat[1] * rhat[1] + qmat[2] * rhat[2],
          qmat[3] * rhat[0] + qmat[4] * rhat[1] + qmat[5] * rhat[2],
          qmat[6] * rhat[0] + qmat[7] * rhat[1] + qmat[8] * rhat[2]};
      float pseudo_axis[3];
      cross3f(rhat, qu, pseudo_axis);
      float pseudo[9];
      stf_outer3f(pseudo_axis, rhat, pseudo);
      float dot = 0.0f;
      float gpseudo[9];
      for (int k = 0; k < 9; ++k) {
        dot += gpd[k] * pseudo[k];
        gpseudo[k] = gpd[k] * weights[c];
      }
      grad_weight[c] += dot;
      float g_axis[3] = {0.0f, 0.0f, 0.0f};
      float gu2[3] = {0.0f, 0.0f, 0.0f};
      add_stf_outer_gradientf(gpseudo, pseudo_axis, rhat, g_axis, gu2);
      float g_u_cross[3];
      float g_qu[3];
      cross3f(qu, g_axis, g_u_cross);
      cross3f(g_axis, rhat, g_qu);
      for (int a = 0; a < 3; ++a) {
        grad_rhat[a] += gu2[a] + g_u_cross[a];
        for (int b = 0; b < 3; ++b) {
          grad_rhat[b] += qmat[3 * a + b] * g_qu[a];
        }
      }

      const float* q_terms = grad_Q_terms + c * kSpinDeg2Count;
      float m2[kSpinDeg2Count];
      fill_spin_monomials2f(rhat, m2);
      grad_weight[c] += dot_spin_termsf(q_terms, m2, kSpinDeg2Count);
      grad_rhat[0] += weights[c] *
          (2.0f * q_terms[0] * rhat[0] + q_terms[3] * rhat[1] +
           q_terms[4] * rhat[2]);
      grad_rhat[1] += weights[c] *
          (2.0f * q_terms[1] * rhat[1] + q_terms[3] * rhat[0] +
           q_terms[5] * rhat[2]);
      grad_rhat[2] += weights[c] *
          (2.0f * q_terms[2] * rhat[2] + q_terms[4] * rhat[0] +
           q_terms[5] * rhat[1]);
    }

    float m2[kSpinDeg2Count];
    float m3[kSpinDeg3Count];
    float m4[kSpinDeg4Count];
    fill_spin_monomials2f(rhat, m2);
    fill_spin_monomialsf(rhat, m3, m4);
    for (int c = 0; c < ChiC; ++c) {
      const float* o_terms = grad_O_terms + c * kSpinDeg3Count;
      const float* o_derivatives = grad_O_derivatives + c * 3 * kSpinDeg2Count;
      const float* h_terms = grad_H_terms + c * kSpinDeg4Count;
      const float* h_derivatives = grad_H_derivatives + c * 3 * kSpinDeg3Count;
      grad_weight[c] += dot_spin_termsf(o_terms, m3, kSpinDeg3Count) +
                        dot_spin_termsf(h_terms, m4, kSpinDeg4Count);
      for (int d = 0; d < 3; ++d) {
        grad_rhat[d] += weights[c] *
            (dot_spin_termsf(
                 o_derivatives + d * kSpinDeg2Count, m2, kSpinDeg2Count) +
             dot_spin_termsf(
                 h_derivatives + d * kSpinDeg3Count, m3, kSpinDeg3Count));
      }
    }

    float weighted_derivative = 0.0f;
    for (int c = 0; c < C; ++c) {
      weighted_derivative += grad_weight[c] * weight_derivatives[c];
    }
    apply_edge_gradients_f32(
        atom,
        neighbor,
        atom_stride,
        dist,
        rhat,
        1.0f,
        weighted_derivative,
        grad_rhat,
        grad_si,
        grad_sj,
        force_soa3,
        mforce_soa3,
        accumulate_virial,
        virial_soa9);
  }
}

__global__ void __launch_bounds__(32, 16) accumulate_spin_chiral_forces(
    int atom_count,
    int atom_stride,
    int struct_dim,
    int num_types,
    int spin_compress,
    int spin_basis_size,
    int spin_l_max,
    float spin_cutoff,
    SimulationBox box,
    const int* __restrict__ types,
    const double* __restrict__ positions_soa3,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ fp,
    const float* __restrict__ descriptor_coefficients,
    int spin_coefficient_offset,
    const float* __restrict__ spin_edge_dx,
    const float* __restrict__ spin_edge_dy,
    const float* __restrict__ spin_edge_dz,
    const float* __restrict__ spin_edge_dist,
    const float* __restrict__ spin_edge_weights,
    const float* __restrict__ spin_edge_weight_derivatives,
    bool use_cached_geometry,
    const float* __restrict__ density_geom_cache,
    const float* __restrict__ chiral_polar_cache,
    const float* __restrict__ chiral_octupoles_raw_cache,
    const float* __restrict__ chiral_hexadecapoles_raw_cache,
    const float* __restrict__ chiral_chirals_cache,
    const float* __restrict__ chiral_pseudodevs_cache,
    double* __restrict__ force_soa3,
    double* __restrict__ mforce_soa3,
    bool accumulate_virial,
    double* __restrict__ virial_soa9) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  (void)num_types;
  (void)spin_basis_size;
  (void)descriptor_coefficients;
  (void)spin_coefficient_offset;
  const int radial_count = nn_radial[atom];
  const int chi_c = spin_compress < 2 ? spin_compress : 2;
  const int base_offset = spin_dim_without_chiral(spin_compress, spin_l_max);
  const int polar_offset = base_offset + chi_c;
  const int pseudo_offset = polar_offset + spin_compress;
  double alpha_chiral_pull[2] = {};
  double alpha_polar_pull[kMaxSpinCompress] = {};
  double alpha_pseudo_pull[kMaxSpinCompress] = {};
  for (int c = 0; c < chi_c; ++c) {
    alpha_chiral_pull[c] = static_cast<double>(
        fp[atom + atom_stride * (struct_dim + base_offset + c)]);
  }
  for (int c = 0; c < spin_compress; ++c) {
    alpha_polar_pull[c] = static_cast<double>(
        fp[atom + atom_stride * (struct_dim + polar_offset + c)]);
    alpha_pseudo_pull[c] = static_cast<double>(
        fp[atom + atom_stride * (struct_dim + pseudo_offset + c)]);
  }

  double geom[kMaxSpinCompress * 9] = {};
  double polar[kMaxSpinCompress * 3] = {};
  double octupoles_raw[2 * kSpinDeg3Count] = {};
  double hexadecapoles_raw[2 * kSpinDeg4Count] = {};

  double chirals[2] = {};
  double pseudodevs[kMaxSpinCompress * 9] = {};
  for (int c = 0; c < spin_compress; ++c) {
    for (int k = 0; k < 3; ++k) {
      polar[idx2(c, k, 3)] =
          static_cast<double>(chiral_polar_cache[atom + atom_stride * (c * 3 + k)]);
    }
    for (int k = 0; k < 9; ++k) {
      geom[idx2(c, k, 9)] =
          static_cast<double>(density_geom_cache[atom + atom_stride * (c * 9 + k)]);
      pseudodevs[idx2(c, k, 9)] = static_cast<double>(
          chiral_pseudodevs_cache[atom + atom_stride * (c * 9 + k)]);
    }
  }
  for (int c = 0; c < chi_c; ++c) {
    for (int k = 0; k < kSpinDeg3Count; ++k) {
      octupoles_raw[c * kSpinDeg3Count + k] = static_cast<double>(
          chiral_octupoles_raw_cache[
              atom + atom_stride * (c * kSpinDeg3Count + k)]);
    }
    for (int k = 0; k < kSpinDeg4Count; ++k) {
      hexadecapoles_raw[c * kSpinDeg4Count + k] = static_cast<double>(
          chiral_hexadecapoles_raw_cache[
              atom + atom_stride * (c * kSpinDeg4Count + k)]);
    }
    chirals[c] =
        static_cast<double>(chiral_chirals_cache[atom + atom_stride * c]);
  }

  double grad_chi[2] = {};
  double grad_polar[kMaxSpinCompress * 3] = {};
  double grad_pseudodev[kMaxSpinCompress * 9] = {};
  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    double rhat[3];
    double dist = 0.0;
    double si[3];
    double sj[3];
    if (use_cached_geometry) {
      load_spin_edge_cached(
          atom,
          neighbor,
          atom_stride,
          slot,
          spins_soa3,
          spin_edge_dx,
          spin_edge_dy,
          spin_edge_dz,
          spin_edge_dist,
          rhat,
          dist,
          si,
          sj);
    } else {
      load_spin_edge(atom, neighbor, atom_stride, box, positions_soa3, spins_soa3,
                     rhat, dist, si, sj);
    }
    if (dist <= 1.0e-12 || dist >= spin_cutoff) {
      continue;
    }
    double weights[kMaxSpinCompress] = {};
    load_spin_edge_weight_cache(
        atom_stride, slot, atom, spin_edge_weights, weights);
    double x[3];
    cross3(si, sj, x);
    const double xu = dot3(x, rhat);
    for (int c = 0; c < chi_c; ++c) {
      const double alpha = alpha_chiral_pull[c];
      grad_chi[c] += alpha * weights[c] * xu;
    }
    for (int c = 0; c < spin_compress; ++c) {
      double axis[3];
      cross3(polar + c * 3, rhat, axis);
      const double alpha_polar = alpha_polar_pull[c];
      double gaxis_polar[3] = {
          alpha_polar * weights[c] * x[0],
          alpha_polar * weights[c] * x[1],
          alpha_polar * weights[c] * x[2]};
      double gp[3];
      cross3(rhat, gaxis_polar, gp);
      for (int d = 0; d < 3; ++d) {
        grad_polar[c * 3 + d] += gp[d];
      }
      const double alpha_pseudo = alpha_pseudo_pull[c];
      double gaxis_pseudo[3] = {
          alpha_pseudo * weights[c] * x[0],
          alpha_pseudo * weights[c] * x[1],
          alpha_pseudo * weights[c] * x[2]};
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          grad_pseudodev[c * 9 + 3 * a + b] += gaxis_pseudo[a] * rhat[b];
        }
      }
    }
  }

  double grad_Q[kMaxSpinCompress * 9] = {};
  double grad_O_terms[2 * kSpinDeg3Count] = {};
  double grad_H_terms[2 * kSpinDeg4Count] = {};
  for (int c = 0; c < chi_c; ++c) {
    const double g = grad_chi[c];
    const double* qmat = geom + c * 9;
    const double* oct_raw = octupoles_raw + c * kSpinDeg3Count;
    const double* hex_raw = hexadecapoles_raw + c * kSpinDeg4Count;
    double* gQ = grad_Q + c * 9;
    double* gO_terms = grad_O_terms + c * kSpinDeg3Count;
    double* gH_terms = grad_H_terms + c * kSpinDeg4Count;
    for (int term = 0; term < kSpinChiralQohCount; ++term) {
      const unsigned short packed = kSpinChiralQohPacked[term];
      const int q = packed >> 8;
      const int o = (packed >> 4) & 0x0f;
      const int h = packed & 0x0f;
      const double scale = g * kSpinChiralQohCoeff[term];
      const double o_value = oct_raw[o];
      const double h_value = hex_raw[h];
      gQ[q] += scale * o_value * h_value;
      const double qscale = scale * qmat[q];
      gO_terms[o] += qscale * h_value;
      gH_terms[h] += qscale * o_value;
    }
  }

  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    double rhat[3];
    double dist = 0.0;
    double si[3];
    double sj[3];
    if (use_cached_geometry) {
      load_spin_edge_cached(
          atom,
          neighbor,
          atom_stride,
          slot,
          spins_soa3,
          spin_edge_dx,
          spin_edge_dy,
          spin_edge_dz,
          spin_edge_dist,
          rhat,
          dist,
          si,
          sj);
    } else {
      load_spin_edge(atom, neighbor, atom_stride, box, positions_soa3, spins_soa3,
                     rhat, dist, si, sj);
    }
    if (dist <= 1.0e-12 || dist >= spin_cutoff) {
      continue;
    }
    double weights[kMaxSpinCompress] = {};
    load_spin_edge_weight_cache(
        atom_stride, slot, atom, spin_edge_weights, weights);
    for (int c = 0; c < spin_compress; ++c) {
      const double* gpd = grad_pseudodev + c * 9;
      const double* qmat = geom + c * 9;
      double qu[3] = {
          qmat[0] * rhat[0] + qmat[1] * rhat[1] + qmat[2] * rhat[2],
          qmat[3] * rhat[0] + qmat[4] * rhat[1] + qmat[5] * rhat[2],
          qmat[6] * rhat[0] + qmat[7] * rhat[1] + qmat[8] * rhat[2]};
      double pseudo_axis[3];
      cross3(rhat, qu, pseudo_axis);
      double gpseudo[9];
      for (int k = 0; k < 9; ++k) {
        gpseudo[k] = gpd[k] * weights[c];
      }
      double g_axis[3] = {0.0, 0.0, 0.0};
      double gu2[3] = {0.0, 0.0, 0.0};
      add_stf_outer_gradient(gpseudo, pseudo_axis, rhat, g_axis, gu2);
      double g_qu[3];
      cross3(g_axis, rhat, g_qu);
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          grad_Q[c * 9 + 3 * a + b] += g_qu[a] * rhat[b];
        }
      }
    }
  }

  double grad_Q_terms[kMaxSpinCompress * kSpinDeg2Count] = {};
  double grad_O_derivatives[2 * 3 * kSpinDeg2Count] = {};
  double grad_H_derivatives[2 * 3 * kSpinDeg3Count] = {};
  for (int c = 0; c < spin_compress; ++c) {
    project_rank2_spin_gradient(grad_Q + c * 9, grad_Q_terms + c * kSpinDeg2Count);
  }
  for (int c = 0; c < chi_c; ++c) {
    fill_spin_term_derivatives(
        3,
        kSpinDeg3Count,
        grad_O_terms + c * kSpinDeg3Count,
        grad_O_derivatives + c * 3 * kSpinDeg2Count);
    fill_spin_term_derivatives(
        4,
        kSpinDeg4Count,
        grad_H_terms + c * kSpinDeg4Count,
        grad_H_derivatives + c * 3 * kSpinDeg3Count);
  }

  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    double rhat[3];
    double dist = 0.0;
    double si[3];
    double sj[3];
    if (use_cached_geometry) {
      load_spin_edge_cached(
          atom,
          neighbor,
          atom_stride,
          slot,
          spins_soa3,
          spin_edge_dx,
          spin_edge_dy,
          spin_edge_dz,
          spin_edge_dist,
          rhat,
          dist,
          si,
          sj);
    } else {
      load_spin_edge(atom, neighbor, atom_stride, box, positions_soa3, spins_soa3,
                     rhat, dist, si, sj);
    }
    if (dist <= 1.0e-12 || dist >= spin_cutoff) {
      continue;
    }
    double weights[kMaxSpinCompress] = {};
    double weight_derivatives[kMaxSpinCompress] = {};
    load_spin_edge_weight_derivative_cache(
        atom_stride,
        slot,
        atom,
        spin_edge_weights,
        spin_edge_weight_derivatives,
        weights,
        weight_derivatives);

    double x[3];
    cross3(si, sj, x);
    double grad_weight[kMaxSpinCompress] = {};
    double grad_rhat[3] = {0.0, 0.0, 0.0};
    double grad_si[3] = {0.0, 0.0, 0.0};
    double grad_sj[3] = {0.0, 0.0, 0.0};
    double gx[3] = {0.0, 0.0, 0.0};
    for (int c = 0; c < chi_c; ++c) {
      const double alpha = alpha_chiral_pull[c];
      const double xu = dot3(x, rhat);
      grad_weight[c] += alpha * xu * chirals[c];
      for (int d = 0; d < 3; ++d) {
        gx[d] += alpha * weights[c] * chirals[c] * rhat[d];
        grad_rhat[d] += alpha * weights[c] * chirals[c] * x[d];
      }
    }
    for (int c = 0; c < spin_compress; ++c) {
      double axis[3];
      cross3(polar + c * 3, rhat, axis);
      const double alpha_polar = alpha_polar_pull[c];
      const double xa = dot3(x, axis);
      grad_weight[c] += alpha_polar * xa;
      double gaxis_polar[3] = {
          alpha_polar * weights[c] * x[0],
          alpha_polar * weights[c] * x[1],
          alpha_polar * weights[c] * x[2]};
      double gu_part[3];
      cross3(gaxis_polar, polar + c * 3, gu_part);
      for (int d = 0; d < 3; ++d) {
        gx[d] += alpha_polar * weights[c] * axis[d];
        grad_rhat[d] += gu_part[d];
      }

      const double alpha_pseudo = alpha_pseudo_pull[c];
      const double* pdev = pseudodevs + c * 9;
      double pseudo_axis[3] = {
          pdev[0] * rhat[0] + pdev[1] * rhat[1] + pdev[2] * rhat[2],
          pdev[3] * rhat[0] + pdev[4] * rhat[1] + pdev[5] * rhat[2],
          pdev[6] * rhat[0] + pdev[7] * rhat[1] + pdev[8] * rhat[2]};
      const double xpa = dot3(x, pseudo_axis);
      grad_weight[c] += alpha_pseudo * xpa;
      double gaxis_pseudo[3] = {
          alpha_pseudo * weights[c] * x[0],
          alpha_pseudo * weights[c] * x[1],
          alpha_pseudo * weights[c] * x[2]};
      for (int d = 0; d < 3; ++d) {
        gx[d] += alpha_pseudo * weights[c] * pseudo_axis[d];
      }
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          grad_rhat[b] += pdev[3 * a + b] * gaxis_pseudo[a];
        }
      }
    }
    double gsi_part[3];
    double gsj_part[3];
    cross3(sj, gx, gsi_part);
    cross3(gx, si, gsj_part);
    for (int d = 0; d < 3; ++d) {
      grad_si[d] += gsi_part[d];
      grad_sj[d] += gsj_part[d];
    }

    for (int c = 0; c < spin_compress; ++c) {
      const double* gp = grad_polar + c * 3;
      grad_weight[c] += dot3(gp, rhat);
      for (int d = 0; d < 3; ++d) {
        grad_rhat[d] += weights[c] * gp[d];
      }

      const double* gpd = grad_pseudodev + c * 9;
      const double* qmat = geom + c * 9;
      double qu[3] = {
          qmat[0] * rhat[0] + qmat[1] * rhat[1] + qmat[2] * rhat[2],
          qmat[3] * rhat[0] + qmat[4] * rhat[1] + qmat[5] * rhat[2],
          qmat[6] * rhat[0] + qmat[7] * rhat[1] + qmat[8] * rhat[2]};
      double pseudo_axis[3];
      cross3(rhat, qu, pseudo_axis);
      double pseudo[9];
      stf_outer3(pseudo_axis, rhat, pseudo);
      double dot = 0.0;
      double gpseudo[9];
      for (int k = 0; k < 9; ++k) {
        dot += gpd[k] * pseudo[k];
        gpseudo[k] = gpd[k] * weights[c];
      }
      grad_weight[c] += dot;
      double g_axis[3] = {0.0, 0.0, 0.0};
      double gu2[3] = {0.0, 0.0, 0.0};
      add_stf_outer_gradient(gpseudo, pseudo_axis, rhat, g_axis, gu2);
      double g_u_cross[3];
      double g_qu[3];
      cross3(qu, g_axis, g_u_cross);
      cross3(g_axis, rhat, g_qu);
      for (int a = 0; a < 3; ++a) {
        grad_rhat[a] += gu2[a] + g_u_cross[a];
        for (int b = 0; b < 3; ++b) {
          grad_rhat[b] += qmat[3 * a + b] * g_qu[a];
        }
      }

      const double* q_terms = grad_Q_terms + c * kSpinDeg2Count;
      double m2[kSpinDeg2Count];
      fill_spin_monomials2(rhat, m2);
      grad_weight[c] += dot_spin_terms(q_terms, m2, kSpinDeg2Count);
      grad_rhat[0] += weights[c] *
          (2.0 * q_terms[0] * rhat[0] + q_terms[3] * rhat[1] +
           q_terms[4] * rhat[2]);
      grad_rhat[1] += weights[c] *
          (2.0 * q_terms[1] * rhat[1] + q_terms[3] * rhat[0] +
           q_terms[5] * rhat[2]);
      grad_rhat[2] += weights[c] *
          (2.0 * q_terms[2] * rhat[2] + q_terms[4] * rhat[0] +
           q_terms[5] * rhat[1]);
    }

    double m2[kSpinDeg2Count];
    double m3[kSpinDeg3Count];
    double m4[kSpinDeg4Count];
    fill_spin_monomials2(rhat, m2);
    fill_spin_monomials(rhat, m3, m4);
    for (int c = 0; c < chi_c; ++c) {
      const double* o_terms = grad_O_terms + c * kSpinDeg3Count;
      const double* o_derivatives = grad_O_derivatives + c * 3 * kSpinDeg2Count;
      const double* h_terms = grad_H_terms + c * kSpinDeg4Count;
      const double* h_derivatives = grad_H_derivatives + c * 3 * kSpinDeg3Count;
      grad_weight[c] += dot_spin_terms(o_terms, m3, kSpinDeg3Count) +
                        dot_spin_terms(h_terms, m4, kSpinDeg4Count);
      for (int d = 0; d < 3; ++d) {
        grad_rhat[d] += weights[c] *
            (dot_spin_terms(
                 o_derivatives + d * kSpinDeg2Count, m2, kSpinDeg2Count) +
             dot_spin_terms(
                 h_derivatives + d * kSpinDeg3Count, m3, kSpinDeg3Count));
      }
    }

    double weighted_derivative = 0.0;
    for (int c = 0; c < spin_compress; ++c) {
      weighted_derivative += grad_weight[c] * weight_derivatives[c];
    }
    apply_edge_gradients(
        atom,
        neighbor,
        atom_stride,
        dist,
        rhat,
        si,
        sj,
        1.0,
        weighted_derivative,
        grad_rhat,
        grad_si,
        grad_sj,
        force_soa3,
        mforce_soa3,
        accumulate_virial,
        virial_soa9);
  }
}

template <bool AtomMajor>
void launch_spin_density_forces_c4_l4(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceWorkspaceView& view,
    bool accumulate_virial,
    int blocks,
    int threads) {
  if constexpr (AtomMajor) {
    if (accumulate_virial) {
      prepare_spin_density_pulls_c4_l4<true><<<blocks, threads>>>(
          atom_count,
          static_cast<int>(view.atom_capacity),
          protocol.struct_descriptor_dim,
          view.spins_soa3,
          view.fp,
          view.spin_density_rho0,
          view.spin_density_l1_rdot,
          view.spin_density_l1_cross,
          view.spin_density_l1_stf,
          view.spin_density_angular2,
          view.spin_density_angular3,
          view.spin_density_angular4,
          view.spin_density_geom,
          view.spin_density_rho0_dot,
          view.spin_density_raw1,
          view.spin_density_raw1_dot,
          view.mforce_soa3);
    }
  } else {
    prepare_spin_density_pulls_c4_l4<AtomMajor><<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.struct_descriptor_dim,
        view.spins_soa3,
        view.fp,
        view.spin_density_rho0,
        view.spin_density_l1_rdot,
        view.spin_density_l1_cross,
        view.spin_density_l1_stf,
        view.spin_density_angular2,
        view.spin_density_angular3,
        view.spin_density_angular4,
        view.spin_density_geom,
        view.spin_density_rho0_dot,
        view.spin_density_raw1,
        view.spin_density_raw1_dot,
        view.mforce_soa3);
  }
  if (accumulate_virial) {
    accumulate_spin_density_forces_c4_l4_pull<AtomMajor><<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.struct_descriptor_dim,
        static_cast<float>(protocol.spin_cutoff_radial),
        view.spins_soa3,
        view.nn_radial,
        view.nl_radial_slot_major,
        view.fp,
        view.spin_edge_dx,
        view.spin_edge_dy,
        view.spin_edge_dz,
        view.spin_edge_dist,
        view.spin_edge_weights,
        view.spin_edge_weight_derivatives,
        view.spin_density_rho0,
        view.spin_density_l1_stf,
        view.spin_density_angular2,
        view.spin_density_angular3,
        view.spin_density_angular4,
        view.spin_density_raw1,
        view.spin_density_rho0_dot,
        view.spin_density_raw1_dot,
        view.force_soa3,
        view.mforce_soa3,
        accumulate_virial,
        view.virial_soa9);
  } else {
    accumulate_spin_density_forces_c4_l4_block_f32<AtomMajor>
        <<<atom_count, threads>>>(
            atom_count,
            static_cast<int>(view.atom_capacity),
            protocol.struct_descriptor_dim,
            static_cast<float>(protocol.spin_cutoff_radial),
            view.spins_soa3,
            view.nn_radial,
            view.nl_radial_slot_major,
            view.fp,
            view.spin_edge_dx,
            view.spin_edge_dy,
            view.spin_edge_dz,
            view.spin_edge_dist,
            view.spin_edge_weights,
            view.spin_edge_weight_derivatives,
            view.spin_density_rho0,
            view.spin_density_l1_rdot,
            view.spin_density_l1_cross,
            view.spin_density_l1_stf,
            view.spin_density_angular2,
            view.spin_density_angular3,
            view.spin_density_angular4,
            view.spin_density_geom,
            view.spin_density_rho0_dot,
            view.spin_density_raw1,
            view.spin_density_raw1_dot,
            view.force_soa3,
            view.mforce_soa3);
  }
}

template <bool AtomMajor>
void launch_spin_chiral_forces_c4_l4(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceWorkspaceView& view,
    bool accumulate_virial,
    int blocks,
    int threads) {
  accumulate_spin_chiral_forces_c4_l4_cached_f32<AtomMajor><<<blocks, threads>>>(
      atom_count,
      static_cast<int>(view.atom_capacity),
      protocol.struct_descriptor_dim,
      static_cast<float>(protocol.spin_cutoff_radial),
      view.spins_soa3,
      view.nn_radial,
      view.nl_radial_slot_major,
      view.fp,
      view.spin_edge_dx,
      view.spin_edge_dy,
      view.spin_edge_dz,
      view.spin_edge_dist,
      view.spin_edge_weights,
      view.spin_edge_weight_derivatives,
      view.spin_density_geom,
      view.spin_chiral_polar,
      view.spin_chiral_octupoles_raw,
      view.spin_chiral_hexadecapoles_raw,
      view.spin_chiral_chirals,
      view.spin_chiral_pseudodevs,
      view.force_soa3,
      view.mforce_soa3,
      accumulate_virial,
      view.virial_soa9);
}

}  // namespace

void build_spin_descriptors_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require(protocol.spin_mode != 0, "spin descriptor requires spin model");
  require(protocol.spin_descriptor_dim > 0, "spin descriptor dimension must be positive");
  require(protocol.spin_descriptor_dim <= 96,
          "spin descriptor kernel supports spin descriptor dim <= 96");
  require(protocol.spin_compress > 0 &&
              protocol.spin_compress <= kMaxSpinCompress,
          "spin descriptor kernel supports spin_compress <= 4");
  require(protocol.spin_basis_size >= 0 &&
              protocol.spin_basis_size + 1 <= kMaxSpinBasis,
          "spin descriptor kernel supports spin_basis_size + 1 <= 8");
  require(protocol.spin_l_max >= 0 && protocol.spin_l_max <= 4,
          "spin descriptor kernel supports spin_l_max <= 4");
  require(all_active(protocol.spin_dof_type_active, protocol.num_types) &&
              all_active(protocol.spin_env_type_active, protocol.num_types),
          "CUDA spin path currently requires all types active for spin dof/env");
  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.types != nullptr, "workspace missing types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.spins_soa3 != nullptr, "workspace missing spins");
  require(view.nn_radial != nullptr, "workspace missing radial neighbor counts");
  require(view.nl_radial_slot_major != nullptr, "workspace missing radial neighbors");
  require(view.descriptors != nullptr, "workspace missing descriptors");
  require(view.spin_density_rho0 != nullptr, "workspace missing spin density rho0");
  require(view.spin_density_raw1 != nullptr, "workspace missing spin density raw1");
  require(view.spin_density_l1_rdot != nullptr,
          "workspace missing spin density l1 rdot");
  require(view.spin_density_l1_cross != nullptr,
          "workspace missing spin density l1 cross");
  require(view.spin_density_l1_stf != nullptr,
          "workspace missing spin density l1 stf");
  require(protocol.spin_l_max < 2 || view.spin_density_angular2 != nullptr,
          "workspace missing spin density angular2");
  require(protocol.spin_l_max < 3 || view.spin_density_angular3 != nullptr,
          "workspace missing spin density angular3");
  require(protocol.spin_l_max < 4 || view.spin_density_angular4 != nullptr,
          "workspace missing spin density angular4");
  require(view.spin_density_geom != nullptr, "workspace missing spin density geom");
  require(view.spin_density_rho0_dot != nullptr,
          "workspace missing spin density rho0 dot");
  require(view.spin_density_raw1_dot != nullptr,
          "workspace missing spin density raw1 dot");
  require(view.spin_edge_dx != nullptr, "workspace missing spin edge dx");
  require(view.spin_edge_dy != nullptr, "workspace missing spin edge dy");
  require(view.spin_edge_dz != nullptr, "workspace missing spin edge dz");
  require(view.spin_edge_dist != nullptr, "workspace missing spin edge dist");
  require(view.spin_edge_weights != nullptr, "workspace missing spin edge weights");
  require(view.spin_edge_weight_derivatives != nullptr,
          "workspace missing spin edge weight derivatives");
  require(protocol.spin_chiral == 0 || view.spin_chiral_polar != nullptr,
          "workspace missing spin chiral polar");
  require(protocol.spin_chiral == 0 || view.spin_chiral_octupoles_raw != nullptr,
          "workspace missing spin chiral octupoles raw");
  require(protocol.spin_chiral == 0 || view.spin_chiral_hexadecapoles_raw != nullptr,
          "workspace missing spin chiral hexadecapoles raw");
  require(protocol.spin_chiral == 0 || view.spin_chiral_chirals != nullptr,
          "workspace missing spin chiral chirals");
  require(protocol.spin_chiral == 0 || view.spin_chiral_pseudodevs != nullptr,
          "workspace missing spin chiral pseudodevs");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
          "model descriptor coefficient buffer is too small");
  const bool use_c4_l4_chiral =
      protocol.spin_compress == 4 && protocol.spin_basis_size == 3 &&
      protocol.spin_l_max == 4 && protocol.spin_chiral != 0;
  if (use_c4_l4_chiral) {
    const int edge_threads = 256;
    const int edge_items = atom_count * protocol.neighbor_capacity_radial;
    const int edge_blocks = (edge_items + edge_threads - 1) / edge_threads;
    if (edge_blocks > 0) {
      precompute_spin_edge_weights_c4<<<edge_blocks, edge_threads>>>(
          atom_count,
          static_cast<int>(view.atom_capacity),
          protocol.neighbor_capacity_radial,
          protocol.num_types,
          static_cast<float>(protocol.spin_cutoff_radial),
          box,
          view.types,
          view.positions_soa3,
          view.nn_radial,
          view.nl_radial_slot_major,
          model_view.descriptor_coefficients,
          static_cast<int>(protocol.ordinary_descriptor_parameter_count),
          view.spin_edge_dx,
          view.spin_edge_dy,
          view.spin_edge_dz,
          view.spin_edge_dist,
          view.spin_edge_weights,
          view.spin_edge_weight_derivatives);
    }
    const int threads = 128;
    const int work_items = atom_count * 4;
    const int blocks = (work_items + threads - 1) / threads;
    if (blocks > 0) {
      if (protocol.neighbor_capacity_radial <= 32) {
        build_spin_primitive_cache_c4_l4_warp<32, true><<<atom_count, 128>>>(
            atom_count,
            static_cast<int>(view.atom_capacity),
            protocol.struct_descriptor_dim,
            static_cast<float>(protocol.spin_cutoff_radial),
            view.spins_soa3,
            view.nn_radial,
            view.nl_radial_slot_major,
            view.spin_edge_dx,
            view.spin_edge_dy,
            view.spin_edge_dz,
            view.spin_edge_dist,
            view.spin_edge_weights,
            view.spin_density_rho0,
            view.spin_density_raw1,
            view.spin_density_l1_rdot,
            view.spin_density_l1_cross,
            view.spin_density_l1_stf,
            view.spin_density_angular2,
            view.spin_density_angular3,
            view.spin_density_angular4,
            view.spin_density_geom,
            view.spin_density_rho0_dot,
            view.spin_density_raw1_dot,
            view.spin_chiral_polar,
            view.spin_chiral_octupoles_raw,
            view.spin_chiral_hexadecapoles_raw,
            view.descriptors);
      } else if (protocol.neighbor_capacity_radial <= kSpinPrimitiveSlots) {
        build_spin_primitive_cache_c4_l4_warp<kSpinPrimitiveSlots, false>
            <<<atom_count, 128>>>(
            atom_count,
            static_cast<int>(view.atom_capacity),
            protocol.struct_descriptor_dim,
            static_cast<float>(protocol.spin_cutoff_radial),
            view.spins_soa3,
            view.nn_radial,
            view.nl_radial_slot_major,
            view.spin_edge_dx,
            view.spin_edge_dy,
            view.spin_edge_dz,
            view.spin_edge_dist,
            view.spin_edge_weights,
            view.spin_density_rho0,
            view.spin_density_raw1,
            view.spin_density_l1_rdot,
            view.spin_density_l1_cross,
            view.spin_density_l1_stf,
            view.spin_density_angular2,
            view.spin_density_angular3,
            view.spin_density_angular4,
            view.spin_density_geom,
            view.spin_density_rho0_dot,
            view.spin_density_raw1_dot,
            view.spin_chiral_polar,
            view.spin_chiral_octupoles_raw,
            view.spin_chiral_hexadecapoles_raw,
            view.descriptors);
      } else {
        build_spin_descriptors_c4_l4_basic<<<blocks, threads>>>(
            atom_count,
            static_cast<int>(view.atom_capacity),
            protocol.struct_descriptor_dim,
            static_cast<float>(protocol.spin_cutoff_radial),
            view.spins_soa3,
            view.nn_radial,
            view.nl_radial_slot_major,
            view.spin_edge_dx,
            view.spin_edge_dy,
            view.spin_edge_dz,
            view.spin_edge_dist,
            view.spin_edge_weights,
            view.spin_density_rho0,
            view.spin_density_raw1,
            view.spin_density_l1_rdot,
            view.spin_density_l1_cross,
            view.spin_density_l1_stf,
            view.spin_density_angular2,
            view.spin_density_angular3,
            view.spin_density_angular4,
            view.spin_density_geom,
            view.spin_density_rho0_dot,
            view.spin_density_raw1_dot,
            view.spin_chiral_polar,
            view.spin_chiral_octupoles_raw,
            view.spin_chiral_hexadecapoles_raw,
            view.descriptors);
      }
      if (protocol.neighbor_capacity_radial <= 32) {
        build_spin_chiral_finalize_c4_l4_f32<true><<<blocks, threads>>>(
            atom_count,
            static_cast<int>(view.atom_capacity),
            protocol.struct_descriptor_dim,
            static_cast<float>(protocol.spin_cutoff_radial),
            view.spins_soa3,
            view.nn_radial,
            view.nl_radial_slot_major,
            view.spin_edge_dx,
            view.spin_edge_dy,
            view.spin_edge_dz,
            view.spin_edge_dist,
            view.spin_edge_weights,
            view.spin_density_geom,
            view.spin_chiral_polar,
            view.spin_chiral_octupoles_raw,
            view.spin_chiral_hexadecapoles_raw,
            view.spin_chiral_chirals,
            view.spin_chiral_pseudodevs,
            view.descriptors);
      } else {
        build_spin_chiral_finalize_c4_l4_f32<false><<<blocks, threads>>>(
            atom_count,
            static_cast<int>(view.atom_capacity),
            protocol.struct_descriptor_dim,
            static_cast<float>(protocol.spin_cutoff_radial),
            view.spins_soa3,
            view.nn_radial,
            view.nl_radial_slot_major,
            view.spin_edge_dx,
            view.spin_edge_dy,
            view.spin_edge_dz,
            view.spin_edge_dist,
            view.spin_edge_weights,
            view.spin_density_geom,
            view.spin_chiral_polar,
            view.spin_chiral_octupoles_raw,
            view.spin_chiral_hexadecapoles_raw,
            view.spin_chiral_chirals,
            view.spin_chiral_pseudodevs,
            view.descriptors);
      }
    }
  } else {
    const int threads = 32;
    const int blocks = (atom_count + threads - 1) / threads;
    if (blocks > 0) {
    build_spin_descriptors<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.struct_descriptor_dim,
        protocol.spin_descriptor_dim,
        protocol.num_types,
        protocol.spin_compress,
        protocol.spin_basis_size,
        protocol.spin_l_max,
        protocol.spin_chiral,
        static_cast<float>(protocol.spin_cutoff_radial),
        box,
        view.types,
        view.positions_soa3,
        view.spins_soa3,
        view.nn_radial,
        view.nl_radial_slot_major,
        model_view.descriptor_coefficients,
        static_cast<int>(protocol.ordinary_descriptor_parameter_count),
        view.spin_density_rho0,
        view.spin_density_raw1,
        view.spin_density_l1_rdot,
        view.spin_density_l1_cross,
        view.spin_density_l1_stf,
        view.spin_density_angular2,
        view.spin_density_angular3,
        view.spin_density_angular4,
        view.spin_density_geom,
        view.spin_density_rho0_dot,
        view.spin_density_raw1_dot,
        view.spin_chiral_polar,
        view.spin_chiral_octupoles_raw,
        view.spin_chiral_hexadecapoles_raw,
        view.spin_chiral_chirals,
        view.spin_chiral_pseudodevs,
        view.descriptors);
    }
  }
  check_cuda(cudaGetLastError(), "build spin descriptors kernel launch failed");
}

void accumulate_spin_onsite_mforces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace) {
  require(protocol.spin_mode != 0, "spin onsite mforce requires spin model");
  require(protocol.spin_descriptor_dim >= 2, "spin descriptor is missing onsite terms");
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.spins_soa3 != nullptr, "workspace missing spins");
  require(view.fp != nullptr, "workspace missing descriptor derivatives");
  require(view.mforce_soa3 != nullptr, "workspace missing mforce output");
  const int threads = 128;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    accumulate_spin_onsite_mforces<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.struct_descriptor_dim,
        view.spins_soa3,
        view.fp,
        view.mforce_soa3);
  }
  check_cuda(cudaGetLastError(), "accumulate spin onsite mforces");
}

void accumulate_spin_scalar_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_virial) {
  require(protocol.spin_mode != 0, "spin scalar force requires spin model");
  require(protocol.spin_compress > 0 &&
              protocol.spin_compress <= kMaxSpinCompress,
          "spin scalar force kernel supports spin_compress <= 4");
  require(protocol.spin_basis_size >= 0 &&
              protocol.spin_basis_size + 1 <= kMaxSpinBasis,
          "spin scalar force kernel supports spin_basis_size + 1 <= 8");
  require(all_active(protocol.spin_dof_type_active, protocol.num_types) &&
              all_active(protocol.spin_env_type_active, protocol.num_types),
          "CUDA spin path currently requires all types active for spin dof/env");
  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.types != nullptr, "workspace missing types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.spins_soa3 != nullptr, "workspace missing spins");
  require(view.nn_radial != nullptr, "workspace missing radial neighbor counts");
  require(view.nl_radial_slot_major != nullptr, "workspace missing radial neighbors");
  require(view.fp != nullptr, "workspace missing descriptor derivatives");
  require(view.force_soa3 != nullptr, "workspace missing forces");
  require(view.mforce_soa3 != nullptr, "workspace missing mforces");
  require(view.virial_soa9 != nullptr, "workspace missing virials");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
          "model descriptor coefficient buffer is too small");
  const bool use_cached_geometry =
      protocol.spin_compress == 4 && protocol.spin_basis_size == 3 &&
      protocol.spin_l_max == 4 && protocol.spin_chiral != 0;
  require(!use_cached_geometry || view.spin_edge_dx != nullptr,
          "workspace missing spin edge dx");
  require(!use_cached_geometry || view.spin_edge_dy != nullptr,
          "workspace missing spin edge dy");
  require(!use_cached_geometry || view.spin_edge_dz != nullptr,
          "workspace missing spin edge dz");
  require(!use_cached_geometry || view.spin_edge_dist != nullptr,
          "workspace missing spin edge dist");
  if (use_cached_geometry && !accumulate_virial) {
    return;
  }
  const int threads = 32;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    accumulate_spin_scalar_forces<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.struct_descriptor_dim,
        protocol.num_types,
        protocol.spin_compress,
        protocol.spin_basis_size,
        static_cast<float>(protocol.spin_cutoff_radial),
        box,
        view.types,
        view.positions_soa3,
        view.spins_soa3,
        view.nn_radial,
        view.nl_radial_slot_major,
        view.fp,
        model_view.descriptor_coefficients,
        static_cast<int>(protocol.ordinary_descriptor_parameter_count),
        view.spin_edge_dx,
        view.spin_edge_dy,
        view.spin_edge_dz,
        view.spin_edge_dist,
        view.spin_edge_weights,
        view.spin_edge_weight_derivatives,
        use_cached_geometry,
        view.force_soa3,
        view.mforce_soa3,
        accumulate_virial,
        view.virial_soa9);
  }
  check_cuda(cudaGetLastError(), "accumulate spin scalar forces");
}

void accumulate_spin_density_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_virial) {
  require(protocol.spin_mode != 0, "spin density force requires spin model");
  require(protocol.spin_compress > 0 &&
              protocol.spin_compress <= kMaxSpinCompress,
          "spin density force kernel supports spin_compress <= 4");
  require(protocol.spin_basis_size >= 0 &&
              protocol.spin_basis_size + 1 <= kMaxSpinBasis,
          "spin density force kernel supports spin_basis_size + 1 <= 8");
  require(protocol.spin_l_max >= 0 && protocol.spin_l_max <= 4,
          "spin density force kernel supports spin_l_max <= 4");
  require(all_active(protocol.spin_dof_type_active, protocol.num_types) &&
              all_active(protocol.spin_env_type_active, protocol.num_types),
          "CUDA spin path currently requires all types active for spin dof/env");
  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.types != nullptr, "workspace missing types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.spins_soa3 != nullptr, "workspace missing spins");
  require(view.nn_radial != nullptr, "workspace missing radial neighbor counts");
  require(view.nl_radial_slot_major != nullptr, "workspace missing radial neighbors");
  require(view.fp != nullptr, "workspace missing descriptor derivatives");
  require(view.force_soa3 != nullptr, "workspace missing forces");
  require(view.mforce_soa3 != nullptr, "workspace missing mforces");
  require(view.virial_soa9 != nullptr, "workspace missing virials");
  require(view.spin_density_rho0 != nullptr, "workspace missing spin density rho0");
  require(view.spin_density_raw1 != nullptr, "workspace missing spin density raw1");
  require(view.spin_density_l1_rdot != nullptr,
          "workspace missing spin density l1 rdot");
  require(view.spin_density_l1_cross != nullptr,
          "workspace missing spin density l1 cross");
  require(view.spin_density_l1_stf != nullptr,
          "workspace missing spin density l1 stf");
  require(protocol.spin_l_max < 2 || view.spin_density_angular2 != nullptr,
          "workspace missing spin density angular2");
  require(protocol.spin_l_max < 3 || view.spin_density_angular3 != nullptr,
          "workspace missing spin density angular3");
  require(protocol.spin_l_max < 4 || view.spin_density_angular4 != nullptr,
          "workspace missing spin density angular4");
  require(view.spin_density_geom != nullptr, "workspace missing spin density geom");
  require(view.spin_density_rho0_dot != nullptr,
          "workspace missing spin density rho0 dot");
  require(view.spin_density_raw1_dot != nullptr,
          "workspace missing spin density raw1 dot");
  require(view.spin_edge_weights != nullptr, "workspace missing spin edge weights");
  require(view.spin_edge_weight_derivatives != nullptr,
          "workspace missing spin edge weight derivatives");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
          "model descriptor coefficient buffer is too small");
  const bool use_cached_geometry =
      protocol.spin_compress == 4 && protocol.spin_basis_size == 3 &&
      protocol.spin_l_max == 4 && protocol.spin_chiral != 0;
  require(!use_cached_geometry || view.spin_edge_dx != nullptr,
          "workspace missing spin edge dx");
  require(!use_cached_geometry || view.spin_edge_dy != nullptr,
          "workspace missing spin edge dy");
  require(!use_cached_geometry || view.spin_edge_dz != nullptr,
          "workspace missing spin edge dz");
  require(!use_cached_geometry || view.spin_edge_dist != nullptr,
          "workspace missing spin edge dist");
  const int threads = 32;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0 && use_cached_geometry) {
    if (protocol.neighbor_capacity_radial <= 32) {
      launch_spin_density_forces_c4_l4<true>(
          protocol, atom_count, view, accumulate_virial, blocks, threads);
    } else {
      launch_spin_density_forces_c4_l4<false>(
          protocol, atom_count, view, accumulate_virial, blocks, threads);
    }
  } else if (blocks > 0) {
    accumulate_spin_density_forces<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.struct_descriptor_dim,
        protocol.num_types,
        protocol.spin_compress,
        protocol.spin_basis_size,
        protocol.spin_l_max,
        static_cast<float>(protocol.spin_cutoff_radial),
        box,
        view.types,
        view.positions_soa3,
        view.spins_soa3,
        view.nn_radial,
        view.nl_radial_slot_major,
        view.fp,
        model_view.descriptor_coefficients,
        static_cast<int>(protocol.ordinary_descriptor_parameter_count),
        view.spin_edge_dx,
        view.spin_edge_dy,
        view.spin_edge_dz,
        view.spin_edge_dist,
        view.spin_edge_weights,
        view.spin_edge_weight_derivatives,
        use_cached_geometry,
        view.spin_density_rho0,
        view.spin_density_raw1,
        view.spin_density_l1_rdot,
        view.spin_density_l1_cross,
        view.spin_density_l1_stf,
        view.spin_density_angular2,
        view.spin_density_angular3,
        view.spin_density_angular4,
        view.spin_density_geom,
        view.spin_density_rho0_dot,
        view.spin_density_raw1_dot,
        view.force_soa3,
        view.mforce_soa3,
        accumulate_virial,
        view.virial_soa9);
  }
  check_cuda(cudaGetLastError(), "accumulate spin density forces");
}

void accumulate_spin_chiral_polar_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_virial) {
  require(protocol.spin_mode != 0, "spin chiral polar force requires spin model");
  require(protocol.spin_chiral != 0, "spin chiral polar force requires chiral model");
  require(protocol.spin_compress > 0 &&
              protocol.spin_compress <= kMaxSpinCompress,
          "spin chiral polar kernel supports spin_compress <= 4");
  require(protocol.spin_basis_size >= 0 &&
              protocol.spin_basis_size + 1 <= kMaxSpinBasis,
          "spin chiral polar kernel supports spin_basis_size + 1 <= 8");
  require(protocol.spin_l_max >= 0 && protocol.spin_l_max <= 4,
          "spin chiral polar kernel supports spin_l_max <= 4");
  require(all_active(protocol.spin_dof_type_active, protocol.num_types) &&
              all_active(protocol.spin_env_type_active, protocol.num_types),
          "CUDA spin path currently requires all types active for spin dof/env");
  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.types != nullptr, "workspace missing types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.spins_soa3 != nullptr, "workspace missing spins");
  require(view.nn_radial != nullptr, "workspace missing radial neighbor counts");
  require(view.nl_radial_slot_major != nullptr, "workspace missing radial neighbors");
  require(view.fp != nullptr, "workspace missing descriptor derivatives");
  require(view.force_soa3 != nullptr, "workspace missing forces");
  require(view.mforce_soa3 != nullptr, "workspace missing mforces");
  require(view.virial_soa9 != nullptr, "workspace missing virials");
  const bool use_cached_geometry =
      protocol.spin_compress == 4 && protocol.spin_basis_size == 3 &&
      protocol.spin_l_max == 4 && protocol.spin_chiral != 0;
  require(view.spin_density_geom != nullptr, "workspace missing spin density geom");
  require(view.spin_chiral_polar != nullptr, "workspace missing spin chiral polar");
  require(view.spin_chiral_octupoles_raw != nullptr,
          "workspace missing spin chiral octupoles raw");
  require(view.spin_chiral_hexadecapoles_raw != nullptr,
          "workspace missing spin chiral hexadecapoles raw");
  require(view.spin_chiral_chirals != nullptr, "workspace missing spin chiral chirals");
  require(view.spin_chiral_pseudodevs != nullptr,
          "workspace missing spin chiral pseudodevs");
  require(view.spin_edge_weights != nullptr, "workspace missing spin edge weights");
  require(view.spin_edge_weight_derivatives != nullptr,
          "workspace missing spin edge weight derivatives");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
          "model descriptor coefficient buffer is too small");
  require(!use_cached_geometry || view.spin_edge_dx != nullptr,
          "workspace missing spin edge dx");
  require(!use_cached_geometry || view.spin_edge_dy != nullptr,
          "workspace missing spin edge dy");
  require(!use_cached_geometry || view.spin_edge_dz != nullptr,
          "workspace missing spin edge dz");
  require(!use_cached_geometry || view.spin_edge_dist != nullptr,
          "workspace missing spin edge dist");
  const int threads = 32;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0 && use_cached_geometry) {
    if (protocol.neighbor_capacity_radial <= 32) {
      launch_spin_chiral_forces_c4_l4<true>(
          protocol, atom_count, view, accumulate_virial, blocks, threads);
    } else {
      launch_spin_chiral_forces_c4_l4<false>(
          protocol, atom_count, view, accumulate_virial, blocks, threads);
    }
  } else if (blocks > 0) {
    accumulate_spin_chiral_forces<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.struct_descriptor_dim,
        protocol.num_types,
        protocol.spin_compress,
        protocol.spin_basis_size,
        protocol.spin_l_max,
        static_cast<float>(protocol.spin_cutoff_radial),
        box,
        view.types,
        view.positions_soa3,
        view.spins_soa3,
        view.nn_radial,
        view.nl_radial_slot_major,
        view.fp,
        model_view.descriptor_coefficients,
        static_cast<int>(protocol.ordinary_descriptor_parameter_count),
        view.spin_edge_dx,
        view.spin_edge_dy,
        view.spin_edge_dz,
        view.spin_edge_dist,
        view.spin_edge_weights,
        view.spin_edge_weight_derivatives,
        use_cached_geometry,
        view.spin_density_geom,
        view.spin_chiral_polar,
        view.spin_chiral_octupoles_raw,
        view.spin_chiral_hexadecapoles_raw,
        view.spin_chiral_chirals,
        view.spin_chiral_pseudodevs,
        view.force_soa3,
        view.mforce_soa3,
        accumulate_virial,
        view.virial_soa9);
  }
  check_cuda(cudaGetLastError(), "accumulate spin chiral polar forces");
}

}  // namespace nep_adapters::cuda_backend
