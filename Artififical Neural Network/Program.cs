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
            MNIST.MnistDataSet dataset = new MNIST.MnistDataSet(@"D:\Code\C#\Artificial Neural Networks\DataSet\Test\t10k-labels.idx1-ubyte", @"D:\Code\C#\Artificial Neural Networks\DataSet\Test\t10k-images.idx3-ubyte");

            NeuralNetwork network = new NeuralNetwork(new Sigmoid(), dataset.trainingData[0].inputData.Length, 16,16, 10);


           network.FeedForward(dataset.trainingData[0]);
           gradientDescent = network.Backpropogate(dataset.trainingData[0]);

            for (int i = 1; i < dataset.trainingData.Length; i++)
            {
                TrainingData training = dataset.trainingData[i];
                network.FeedForward(training);
                gradientDescent += network.Backpropogate(training);
                Console.WriteLine("Finished: " + i);
            }
            gradientDescent /= dataset.trainingData.Length;

            network.Save("D:/Code/C#/Artificial Neural Networks/TestResults/Test.txt");

            Console.Read();
        }
    }
}
