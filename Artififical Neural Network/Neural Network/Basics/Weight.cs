using Newtonsoft.Json;

namespace NeuralNetworks
{
    public class Weight : IBackpropogatable
    {
        public readonly int layerIndex;
        public double weight;

        public Connection connection;

        /// <summary>
        /// Constructor for Weight class. That generates random weight between -1 and 1.
        /// </summary>
        /// <param name="layerIndex"></param>
        public Weight(int layerIndex, Neuron startNeuron, Neuron endNeuron)
        {
            this.layerIndex = layerIndex;

            weight = Randomizer.Range(-1, 1);
            connection = new Connection(startNeuron, endNeuron);
        }

        [JsonConstructor]
        public Weight(int layerIndex, Connection connection, double weight)
        {
            this.layerIndex = layerIndex;
            this.weight = weight;
            this.connection = connection;
        }

        /// <summary>
        /// Back propogate the weight
        /// </summary>
        /// <returns></returns>
        public void BackPropogate(GradientDescent gradient)
        {
            double value = connection.startNeuron.Activation * connection.endNeuron.DerivativeActivation * connection.endNeuron.DerivativeCost;
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
