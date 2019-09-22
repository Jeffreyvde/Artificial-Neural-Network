using System;

namespace NeuralNetworks
{
    public class Weight : IBackpropogatable
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

            weight = Randomizer.GetRandomNumber(-1, 1);
            connections = new Connection(startNeuron, endNeuron);
        }

        /// <summary>
        /// Back propogate the weight
        /// </summary>
        /// <returns></returns>
        public void BackPropogate(GradientDescent gradient)
        {
            double value = connections.startNeuron.activation * connections.endNeuron.derivativeActivation * connections.endNeuron.derivativeCost;
            gradient.Add(value, this);
        }
        
        /// <summary>
        /// Apply the gradient decent step
        /// </summary>
        /// <param name="steo"></param>
        public void ApplyGradientDecentStep(double step)
        {
            weight += step;
        }

    }
}
