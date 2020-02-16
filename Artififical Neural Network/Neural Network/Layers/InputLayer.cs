using NeuralNetwork.Neurons;

namespace NeuralNetwork.Layers
{
    public class InputLayer : Layer<InputNeuron>
    {
        /// <summary>
        /// Initialize from size
        /// </summary>
        /// <param name="size"></param>
        /// <param name="activationFunction"></param>
        public InputLayer(int size)
        {
            Neurons = new InputNeuron[size];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new InputNeuron();
            }
        }

        /// <summary>
        /// Initialize from training data
        /// </summary>
        /// <param name="startData"></param>
        /// <param name="activationFunction"></param>
        public InputLayer(double[] startData)
        {
            //TODO check null
            Neurons = new InputNeuron[startData.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new InputNeuron(startData[i]);
            }
        }

        public void SetInput(double[] startData)
        {
            //TODO check null and check size is equal to neurons length
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].SetActivation(startData[i]);
            }
        }
    }
}
