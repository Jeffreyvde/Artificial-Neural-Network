using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class TrainingManager
    {
        private Batch[,] batches;
        private readonly TrainingData[] data;
        private readonly int iterations, batchesPerIterations, batchSize, iterationSize;


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

            this.data = data;
            this.batchesPerIterations = batchesPerIterations;
            this.iterations = iterations;
            this.learningRate = learningRate;
            this.testData = testData;
            this.batchSize = batchSize;

            iterationSize = batchesPerIterations * data[0].inputData.Length;

            GenerateBatches();
            this.neuralNetwork = neuralNetwork;
        }

        
        /// <summary>
        /// Generate new batches 
        /// </summary>
        private void GenerateBatches()
        {
            for (int i = 0; i < iterations; i++)
            {
                for (int j = 0; j < batchesPerIterations; j++)
                {
                    batches[i, j] = new Batch(data, batchSize);
                }
            }
        }

        /// <summary>
        /// Train until a specific required percentage
        /// </summary>
        /// <param name="requiredPercentage"></param>
        public void TrainUntil(float requiredPercentage, string savepath = "")
        {
            do
            {
                GenerateBatches();
                Train();
                if (savepath != "")
                    neuralNetwork.Save(savepath);

            } while (Test() < requiredPercentage);
        }

        /// <summary>
        /// Train the neural network via batches
        /// </summary>
        /// <param name="network"></param>
        public void Train()
        {

            for (int i = 0; i < iterations; i++)
            {
                GradientDescent gradientDescent = batches[0, 0].Run(neuralNetwork);
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
        }


        /// <summary>
        /// Test the neural network
        /// </summary>
        private float Test()
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
            return percentage;
        }
    }
}

