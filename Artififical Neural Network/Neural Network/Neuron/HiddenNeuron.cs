using System;
using NeuralNetwork.Activations;
using NeuralNetwork.Backpropogation;
using NeuralNetwork.Utilities;

namespace NeuralNetwork.Neurons
{
    [Serializable]
    public class HiddenNeuron : Neuron
    {
        /// <summary>
        /// Default constructor for the hidden neuron
        /// </summary>
        /// <param name="activationFunction"></param>
        public HiddenNeuron(IActivation activationFunction) : base(activationFunction) { }

        /// <summary>
        /// Constructor with random is injected
        /// </summary>
        /// <param name="random">The random class to be used</param>
        /// <param name="activationFunction">The activation function to be used</param>
        public HiddenNeuron(IRandom random, IActivation activationFunction) : base(random, activationFunction) { }

        /// <summary>
        /// Back propagate this hidden neuron
        /// </summary>
        /// <param name="descent"></param>
        public override void BackPropagate(GradientDescent descent)
        {
            if(descent == null)
                throw new ArgumentNullException(nameof(descent));

            for (int i = 0; i < ForwardConnections.Length; i++)
            {
                Neuron nextNeuron = (Neuron)ForwardConnections[i].EndNeuron;
                DerivativeCost += ForwardConnections[i].Weight * nextNeuron.DerivativeActivation * nextNeuron.DerivativeActivation;
            }
            DerivativeCost /= ForwardConnections.Length;
            base.BackPropagate(descent);

            descent.Add(DerivativeActivation * DerivativeCost, this);
        }
    }
}
