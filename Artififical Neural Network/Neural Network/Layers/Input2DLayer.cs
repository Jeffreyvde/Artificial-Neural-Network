namespace NeuralNetwork.Layers
{
    [System.Serializable]
    public class Input2DLayer : InputLayer
    {
        private readonly uint length, width;

        /// <summary>
        /// Constructor for the 2D Input layer class
        /// </summary>
        /// <param name="length">Length of the layer</param>
        /// <param name="width">Width of the layer</param>
        public Input2DLayer(uint length, uint width) : base(length * width)
        {
            this.length = length;
            this.width = width;
        }

        /// <summary>
        /// Set the input layers activations
        /// </summary>
        /// <param name="input">Make sure that this value is the same length and width</param>
        public void SetInput(double[,] input)
        {
            if (input == null) 
                throw new System.ArgumentNullException(nameof(input));

            double[] startData = new double[input.Length];
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    startData[i * width + j] = input[i, j];
                }
            }
            SetInput(startData);
        }
    }
}
