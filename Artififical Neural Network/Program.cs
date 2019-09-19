using System;
using NeuralNetworks;

namespace Artififical_Neural_Network
{
    class Program
    {
        static void Main()
        {
            NeuralNetwork network = new NeuralNetwork(new Sigmoid(),3, 3, 3);
            TrainingData trainingData = new TrainingData(new double[]{ 1, 0, 1 }, 0);

            network.Train(trainingData);

            Console.Read();
        }
    }
}
