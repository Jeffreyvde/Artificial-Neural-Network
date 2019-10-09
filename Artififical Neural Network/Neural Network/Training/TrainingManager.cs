using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class TrainingManager
    {
        private readonly Batch[,] batches;
        private readonly int iterations, batchesPerIterations, iterationSize;


        private TrainingData[] testData;
        private NeuralNetwork neuralNetwork;

        private readonly double learningRate;


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
            iterationSize = batchesPerIterations * data[0].inputData.Length;

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
        public void Train(string path)
        {
            while (true)
            {
                for (int i = 0; i < iterations; i++)
                {
                    GradientDescent gradientDescent = batches[0,0].Run(neuralNetwork);
                    List<Task> TaskList = new List<Task>();
                    for (int j = i; j < batchesPerIterations; j++)
                    {
                        if (i == 0 && j == 0) continue;

                        Batch batch = batches[i, j];
                        Task task = Task.Run(() => gradientDescent += batch.Run(neuralNetwork));
                        TaskList.Add(task);
                    }
                    Task.WaitAll(TaskList.ToArray());
                    gradientDescent.Apply(learningRate / iterationSize);
                    Console.WriteLine("Finished iteration: " + i);
                }
                Test();
                neuralNetwork.Save(path);
            }
        }

        /// <summary>
        /// Test the neural network
        /// </summary>
        private void Test()
        {
            float correctThings = 0;
            for (int i = 0; i < testData.Length; i++)
            {
                TrainingData training = testData[i];
                neuralNetwork.FeedForward(training);
                if (neuralNetwork.IsNeuralNetworkCorrect(training.correctOutputNeuron))
                {
                    correctThings++;
                }
            }
            float percentage = correctThings / testData.Length * 100f;
            Console.WriteLine("Network Percentage: " + percentage);
        }
    }
}

