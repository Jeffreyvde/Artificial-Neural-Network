using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class TrainingManager
    {
        private Batch[,] batches;
        private int iterations, batchesPerIterations;


        private TrainingData[] testData;
        private NeuralNetwork neuralNetwork;

        private double learningRate;


        /// <summary>
        /// Train a neural network
        /// </summary>
        /// <param name="iterations"></param>
        /// <param name="batches"></param>
        public TrainingManager(int iterations, TrainingData[] data, TrainingData[] testData, int batchesPerIterations, int batchSize, NeuralNetwork neuralNetwork, double learningRate = 3)
        {
            batches = new Batch[iterations, batchesPerIterations];
            this.batchesPerIterations = batchesPerIterations;
            this.iterations = iterations;
            this.learningRate = learningRate;
            this.testData = testData;

            for (int i = 0; i < iterations; i++)
            {
                for (int j = 0; j < batchesPerIterations; j++)
                {
                    batches[i, j] = new Batch(data, batchSize);
                }
            }
            this.neuralNetwork = neuralNetwork;
        }

        /// <summary>
        /// Train the neural network via batches
        /// </summary>
        /// <param name="network"></param>
        public void Train()
        {
            while (true)
                for (int i = 0; i < iterations; i++)
                {
                    for (int j = 0; j < batchesPerIterations; j++)
                    {
                        Batch batch = batches[i, j];
                        batch.Run(neuralNetwork, learningRate);
                    }
                    Test();
                    neuralNetwork.Save("D:/Code/C#/Artificial Neural Networks/TestResults/Test.txt");
                    //List<Task> TaskList = new List<Task>();
                    //for (int j = 0; j < batchesPerIterations; j++)
                    //{
                    //    Batch batch = batches[i, j];
                    //    Task task = Task.Run(() => batch.Run(neuralNetwork));
                    //    TaskList.Add(task);
                    //}
                    //Task.WaitAll(TaskList.ToArray());
                }
        }


        public void Test()
        {
            float correctThings = 0;
            for (int i = 0; i < testData.Length; i++)
            {
                TrainingData training = testData[i];
                neuralNetwork.FeedForward(training);
                if (neuralNetwork.IsNeuralNetworkCorrect(training.correctOutputNeuron))
                {
                    //Console.WriteLine("Correct");
                    correctThings++;
                    //Console.WriteLine(neuralNetwork.CalculateCost(training.correctOutputNeuron));
                }
            }
            float percentage = correctThings / testData.Length * 100f;
            Console.WriteLine("Network Percentage: " + percentage);
        }
    }
}
