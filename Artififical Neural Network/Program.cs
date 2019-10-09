using System;
using System.IO;
using NeuralNetworks;

namespace Artififical_Neural_Network
{
    class Program
    {
        static void Main()
        {


            MNIST.MnistDataSet dataset = new MNIST.MnistDataSet(@"D:\Code\C#\Artificial Neural Networks\DataSet\Training\train-labels.idx1-ubyte", @"D:\Code\C#\Artificial Neural Networks\DataSet\Training\train-images.idx3-ubyte");
            MNIST.MnistDataSet testing = new MNIST.MnistDataSet(@"D:\Code\C#\Artificial Neural Networks\DataSet\Test\t10k-labels.idx1-ubyte", @"D:\Code\C#\Artificial Neural Networks\DataSet\Test\t10k-images.idx3-ubyte");

            NeuralNetwork network = new NeuralNetwork(new Sigmoid(), dataset.trainingData[0].inputData.Length, 30, 10);
            //NeuralNetwork network = NeuralNetwork.Load("D:/Code/C#/Artificial Neural Networks/TestResults/Test.txt", new Sigmoid());

            TrainingManager manager = new TrainingManager(40, dataset.trainingData, testing.trainingData, 10, 10, network, 3);
            manager.Train();

            Console.Read();
        }
    }
}
