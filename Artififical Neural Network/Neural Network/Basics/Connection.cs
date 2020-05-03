using NeuralNetwork.Backpropogation;
using NeuralNetwork.Utilities;

namespace NeuralNetwork.Neurons
{
    [System.Serializable]
    public class Connection
    {
        public BaseNeuron StartNeuron { get; private set; }
        public BaseNeuron EndNeuron { get; private set; }

        public double Weight => weight.Value;

        private readonly Weight weight;

        /// <summary>
        /// Constructor for a connection between neurons
        /// </summary>
        /// <param name="startNeuron"></param>
        /// <param name="endNeuron"></param>
        public Connection(BaseNeuron startNeuron, BaseNeuron endNeuron) : this(startNeuron, endNeuron, new RandomRange())
        {
        }

        /// <summary>
        /// Constructor for a connection between neurons
        /// </summary>
        /// <param name="startNeuron"></param>
        /// <param name="endNeuron"></param>
        public Connection(BaseNeuron startNeuron, BaseNeuron endNeuron, IRandom random)
        {
            StartNeuron = startNeuron;
            EndNeuron = endNeuron;

            weight = new Weight(random);
        }

        public void BackPropagate(GradientDescent descent)
        {
            weight.BackPropagate(this, descent);
        }
    }
}
