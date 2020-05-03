using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Utilities
{
    public interface IRandom
    {
        double Range(double minimum, double maximum);
    }
}
