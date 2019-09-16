using System;

namespace NeuralNetworks
{
    public class Weight
    {
        public readonly int layerIndex;
        public double weight;

        public Connection connections;

        /// <summary>
        /// Constructor for Weight class. That generates random weight between 0 and 1.
        /// </summary>
        /// <param name="layerIndex"></param>
        /// <param name="random"></param>
        public Weight(int layerIndex, Neuron startNeuron, Neuron endNeuron, Random random)
        {
            this.layerIndex = layerIndex;

            weight = (float)random.NextDouble();
            connections = new Connection(startNeuron, endNeuron);
        }
    }
}
