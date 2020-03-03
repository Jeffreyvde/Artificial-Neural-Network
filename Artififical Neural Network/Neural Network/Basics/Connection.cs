using NeuralNetwork.Backpropogation;

namespace NeuralNetwork.Neurons
{
    public class Connection
    {
        public BaseNeuron StartNeuron { get; private set; }
        public BaseNeuron EndNeuron { get; private set; }

        public double Weight { get { return weight.weight; } }

        private readonly Weight weight;

        /// <summary>
        /// Constructor for a connection between Neurons
        /// </summary>
        /// <param name="startNeuron"></param>
        /// <param name="endNeuron"></param>
        public Connection(BaseNeuron startNeuron, BaseNeuron endNeuron)
        {
            StartNeuron = startNeuron;
            EndNeuron = endNeuron;

            weight = new Weight();
        }

        public void BackPropogate(GradientDescent descent)
        {
            weight.BackPropogate(this, descent);
        }
    }
}
