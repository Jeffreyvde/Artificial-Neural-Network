using System;

namespace NeuralNetwork
{
    public class Neuron
    {
        public readonly int layerIndex, layerRow;

        public float activation;

        public float bias;
        public float weightedSum;

        /// <summary>
        /// Constructor for Neuron class. That generates random bias between 0 and 1.
        /// </summary>
        /// <param name="layerIndex"></param>
        /// <param name="random"></param>
        public Neuron(int layerIndex, int layerRow, Random random)
        {
            this.layerIndex = layerIndex;
            this.layerRow = layerRow;

            bias = (float)random.NextDouble();
        }

        /// <summary>
        /// Calculate the activation of this neuron
        /// </summary>
        /// <param name="sum"></param>
        public void CalculateActivation(float sum)
        {

        }
    }
}
