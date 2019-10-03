using System;
using System.IO;
using NeuralNetworks;

namespace Artififical_Neural_Network
{
    class Program
    {

        private static NeuralNetwork network; /*NeuralNetwork.Load("D:/Code/C#/Artificial Neural Networks/TestResults/Test.txt", new Sigmoid());*/ //new NeuralNetwork(new Sigmoid(), dataset.trainingData[0].inputData.Length, 16, 16, 10);

        private static GradientDescent gradientDescent;

        static void Main()
        {
            for (int i = 0; i < 1; i++)
            {
                Train();
            }
            Test();
            Console.Read();
        }

        private static void Train()
        {
            //MNIST.MnistDataSet dataset = new MNIST.MnistDataSet(@"D:\Code\C#\Artificial Neural Networks\DataSet\Training\train-labels.idx1-ubyte", @"D:\Code\C#\Artificial Neural Networks\DataSet\Training\train-images.idx3-ubyte");
            MNIST.MnistDataSet dataset = new MNIST.MnistDataSet(@"D:\Code\C#\Artificial Neural Networks\DataSet\Test\t10k-labels.idx1-ubyte", @"D:\Code\C#\Artificial Neural Networks\DataSet\Test\t10k-images.idx3-ubyte");

            network = new NeuralNetwork(new Sigmoid(), dataset.trainingData[0].inputData.Length, 16, 16, 10);
            for (int test = 0; test < 40; test++)
            {
                network.FeedForward(dataset.trainingData[0]);
                gradientDescent = network.Backpropogate(dataset.trainingData[0]);

                for (int i = 1; i < dataset.trainingData.Length; i++)
                {
                    TrainingData training = dataset.trainingData[i];
                    network.FeedForward(training);
                    gradientDescent += network.Backpropogate(training);
                }
                gradientDescent /= dataset.trainingData.Length - 1;
                gradientDescent.Apply();

                network.Save("D:/Code/C#/Artificial Neural Networks/TestResults/Test.txt");
                Test();
            }
        }
        private static void Test()
        {
            MNIST.MnistDataSet dataset = new MNIST.MnistDataSet(@"D:\Code\C#\Artificial Neural Networks\DataSet\Test\t10k-labels.idx1-ubyte", @"D:\Code\C#\Artificial Neural Networks\DataSet\Test\t10k-images.idx3-ubyte");

            //NeuralNetwork network = NeuralNetwork.Load("D:/Code/C#/Artificial Neural Networks/TestResults/Test.txt", new Sigmoid());

            float length = dataset.trainingData.Length;
            float correctThings = 0;

            for (int i = 0; i < dataset.trainingData.Length; i++)
            {
                TrainingData training = dataset.trainingData[i];
                network.FeedForward(training);
                //Console.WriteLine("Cost: " + network.CalculateCost(training.correctOutputNeuron));
                if (network.IsNeuralNetworkCorrect(training.correctOutputNeuron))
                {
                    //Console.WriteLine("Correct");
                    //Console.WriteLine(training);
                    correctThings++;
                }
            }
            float percentage = correctThings / length * 100f;
            Console.WriteLine("Network Percentage: " + percentage);
        }
    }
}
