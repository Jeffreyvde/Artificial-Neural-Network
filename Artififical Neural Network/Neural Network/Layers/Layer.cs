using Newtonsoft.Json;
using NeuralNetwork.Neurons;

namespace NeuralNetwork.Layers
{
    public abstract class Layer
    {
        public BaseNeuron[] Neurons { get; protected set; }

        /// <summary>
        /// Default constructor for layers
        /// </summary>
        protected Layer() { }

        #region Initialization
        [JsonConstructor()]
        public Layer(BaseNeuron[] neurons)
        {
            Neurons = neurons;
        }

        /// <summary>
        /// Generate weights for this layer
        /// </summary>
        /// <param name="previousNeurons">previous neurons to create Connection</param>
        public void ConnectNeurons(BaseNeuron[] nextLayer, BaseNeuron[] previousLayer)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].EstablishConnections(nextLayer, previousLayer);
            }
        }
        #endregion
        #region Training

        /// <summary>
        /// Train this layer
        /// </summary>
        /// <param name="previousNeurons"></param>
        public void FeedForward()
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].FeedForward();
            }
        }

        #endregion
        #region Backpropogation

        /// <summary>
        /// Backpropogate this output layer
        /// </summary>
        /// <param name="correctOutputNeuron"></param>
        public void BackPropogate(GradientDescent gradient)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].BackPropogate(gradient);
            }
        }
        #endregion
    }
}
