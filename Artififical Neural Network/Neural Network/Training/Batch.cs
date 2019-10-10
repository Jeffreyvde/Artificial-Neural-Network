﻿namespace NeuralNetworks
{
    public class Batch
    {
        public TrainingData[] data;

        /// <summary>
        /// Create randomized Batch
        /// </summary>
        /// <param name="allData"></param>
        /// <param name="size"></param>
        public Batch(TrainingData[] allData, int size)
        {
            data = new TrainingData[size];
            for (int i = 0; i < size; i++)
            {
                data[i] = allData[Randomizer.random.Next(0, allData.Length)];
            }

        }

        /// <summary>
        /// Run a neural network on batch
        /// </summary>
        /// <param name="network"></param>
        public GradientDescent Run(NeuralNetwork network)
        {
            network.FeedForward(data[0]);
            GradientDescent gradientDescent = network.Backpropogate(data[0]);

            for (int i = 1; i < data.Length; i++)
            {
                TrainingData training = data[i];
                network.FeedForward(training);
                gradientDescent += network.Backpropogate(training);
            }
            gradientDescent /= data.Length - 1;
            return gradientDescent;
        }
    }
}