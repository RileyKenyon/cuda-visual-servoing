///-----------------------------------------------------------------------------
/// @file logging.hpp
///
/// @author Riley Kenyon (rike2277@colorado.edu)
/// @brief
///
/// @date 2024-01-27
///-----------------------------------------------------------------------------
#ifndef LOGGING_HPP
#define LOGGING_HPP
#include <fstream>

void log_to_file(const std::string &fname, const unsigned int *data, int width, int height) {
  std::ofstream dataFile;
  dataFile.open(fname);
  for (int k = 0; k < 3; k++) {
    const int offset = 100;
    for (int j = offset * (k + 1) * width; j < ((offset * (k + 1)) + 1) * width; j++) {
      int output = data[j] - '0';
      dataFile << output;
      if (j != ((offset * (k + 1)) + 1) * width - 1) { // last element in row
        dataFile << ",";
      }
      // data[j] = 0;
    }
    dataFile << "\n";
  }
  dataFile.close();
}

#endif // LOGGING_HPP
