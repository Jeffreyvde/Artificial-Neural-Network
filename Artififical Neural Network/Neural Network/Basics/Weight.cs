using System;
using NeuralNetwork.Backpropogation;
using NeuralNetwork.Utilities;

namespace NeuralNetwork.Neurons
{
    [Serializable]
    public class Weight : IBackpropogatable
    {
        public double Value { get; private set; }

        /// <summary>
        /// Constructor for Weight class. That generates random Value between -1 and 1.
        /// </summary>
        public Weight() : this(new RandomRange())
        {
        }

        /// <summary>
        /// Constructor for Weight class. That generates random Value between -1 and 1.
        /// </summary>
        /// <param name="random">Random value to be used for testing</param>
        public Weight(IRandom random)
        {
            if (random == null)
                throw new ArgumentNullException(nameof(random));
            Value = random.Range(-1, 1);
        }

        /// <summary>
        /// Back propogate the Value
        /// </summary>
        /// <returns></returns>
        public void BackPropagate(Connection connection, GradientDescent gradient)
        {
            if(gradient == null)
                throw new ArgumentNullException(nameof(gradient));
            if(connection == null)
                throw new ArgumentNullException(nameof(connection));

            double value = connection.StartNeuron.Activation * ((Neuron)connection.EndNeuron).DerivativeActivation * ((Neuron)connection.EndNeuron).DerivativeCost;
            gradient.Add(value, this);
        }

        /// <summary>
        /// Apply the gradient decent step
        /// </summary>
        /// <param name="step">The step that needs to be applied</param>
        /// <param name="learningRate">The learning rate that needs to be applied</param>
        public void ApplyGradientDecentStep(double step, double learningRate)
        {
            Value -= learningRate * step;
        }

    }
}
