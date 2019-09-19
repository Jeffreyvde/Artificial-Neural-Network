using System;

namespace NeuralNetworks
{
    public class Weight
    {
        public readonly int layerIndex;
        public double weight;

        public Connection connections;

        /// <summary>
        /// Constructor for Weight class. That generates random weight between -1 and 1.
        /// </summary>
        /// <param name="layerIndex"></param>
        public Weight(int layerIndex, Neuron startNeuron, Neuron endNeuron)
        {
            this.layerIndex = layerIndex;

            weight = Randomizer.GetRandomNumber(-1,1);
            connections = new Connection(startNeuron, endNeuron);
        }
    }
}
