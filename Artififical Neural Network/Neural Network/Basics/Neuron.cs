using System;

namespace NeuralNetworks
{
    public class Neuron
    {
        public readonly int layerIndex, layerRow;

        public double activation;

        public double bias;
        public double weightedSum;

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
        /// Constructor for Neuron class. That already has an activation.
        /// </summary>
        /// <param name="layerIndex"></param>
        /// <param name="random"></param>
        public Neuron(int layerIndex, int layerRow, double activation)
        {
            this.layerIndex = layerIndex;
            this.layerRow = layerRow;

            this.activation = activation;
        }

    }
}
