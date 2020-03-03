using NeuralNetwork.Neurons;
using NeuralNetwork.Utilities;

namespace NeuralNetwork
{
    public class Weight : IBackpropogatable
    {
        public double weight { get; private set; }

        /// <summary>
        /// Constructor for Weight class. That generates random weight between -1 and 1.
        /// </summary>
        /// <param name="layerIndex"></param>
        public Weight()
        {
            weight = Random.Range(-1, 1);
        }

        /// <summary>
        /// Back propogate the weight
        /// </summary>
        /// <returns></returns>
        public void BackPropogate(Connection connection, GradientDescent gradient)
        {
            double value = connection.StartNeuron.Activation * ((Neuron)connection.EndNeuron).DerivativeActivation * ((Neuron)connection.EndNeuron).DerivativeCost;
            gradient.Add(value, this);
        }

        /// <summary>
        /// Apply the gradient decent step
        /// </summary>
        /// <param name="steo"></param>
        public void ApplyGradientDecentStep(double step, double learningRate)
        {
            weight -= learningRate * step;
        }

    }
}
