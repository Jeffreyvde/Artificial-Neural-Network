using System;

namespace NeuralNetwork
{
    public class Weight
    {
        public readonly int layerIndex;
        public float weight;

        public Connection connection;

        /// <summary>
        /// Constructor for Weight class. That generates random weight between 0 and 1.
        /// </summary>
        /// <param name="layerIndex"></param>
        /// <param name="random"></param>
        public Weight(int layerIndex, Random random, Neuron startNeuron, Neuron endNeuron)
        {
            this.layerIndex = layerIndex;

            weight = (float)random.NextDouble();
            connection = new Connection(startNeuron, endNeuron);
        }
    }
}
