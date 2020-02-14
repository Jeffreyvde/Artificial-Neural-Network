using Newtonsoft.Json;
using NeuralNetwork.Neurons;

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
            weight = Randomizer.Range(-1, 1);
        }

        [JsonConstructor]
        public Weight(double weight)
        {
            this.weight = weight;
        }

        /// <summary>
        /// Back propogate the weight
        /// </summary>
        /// <returns></returns>
        public void BackPropogate(Connection connection, GradientDescent gradient)
        {
            double value = connection.StartNeuron.Activation * connection.EndNeuron.DerivativeActivation * connection.EndNeuron.derivativeCost;
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
