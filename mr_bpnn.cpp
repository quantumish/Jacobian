#include "bpnn.hpp"
#include "utils.hpp"

class ParallelNetwork : Network
{
public:
  std::function<pair*(pair)> map;
  std::function<pair*(pair*)> reduce;
  //   See if this compiles.
};

int main()
{
  return 0; //   Wow.
}
