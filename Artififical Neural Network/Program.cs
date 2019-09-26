using System;
using System.IO;
using NeuralNetworks;

namespace Artififical_Neural_Network
{
    class Program
    {
        private static GradientDescent gradientDescent;

        static void Main()
        {
            NeuralNetwork network = NeuralNetwork.Load("D:/Code/C#/Artificial Neural Networks/TestResults/Test.txt"); // new NeuralNetwork(new Sigmoid(), 3, 3, 3);
            TrainingData trainingData = new TrainingData(new double[] { 1, 0.1, 1 }, 0);

            network.FeedForward(trainingData);
            gradientDescent = network.Backpropogate(trainingData);
            gradientDescent.Apply();

            network.Save("D:/Code/C#/Artificial Neural Networks/TestResults/Test.txt");

            Console.Read();
        }
    }
}
