namespace NeuralNetwork.Layers
{
    public class Input2DLayer : InputLayer
    {
        private readonly int length, width;

        /// <summary>
        /// Constructor for the 2D Input layer class
        /// </summary>
        /// <param name="length"></param>
        /// <param name="width"></param>
        public Input2DLayer(int length, int width) : base(length * width)
        {
            this.length = length;
            this.width = width;
        }
        
        /// <summary>
        /// Set the input layers activations
        /// </summary>
        /// <param name="input"></param>
        public void SetInput(double[,] input)
        {
            if (input == null) throw new System.NullReferenceException("Startdata can not be null: please check 2D Input layer's SetInput function");

            double[] startData = new double[input.Length];
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    startData[i * width + j] = input[i, j];
                }
            }
            SetInput(startData);
            inputType = typeof(double[,]);
        }
    }
}
