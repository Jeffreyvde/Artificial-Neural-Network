using NeuralNetwork.Utilities;

namespace NeuralNetwork.Tests.Mocks
{
    public class MockRandom : IRandom
    {
        public double Range(double minimum, double maximum)
        {
            return minimum == -1 && maximum == 1 ? 1 : 0;
        }
    }
}
