using System;

namespace NeuralNetwork.Utilities
{
    public class RandomRange : IRandom
    {
        private readonly System.Random random;

        /// <summary>
        /// Create the default random class
        /// </summary>
        public RandomRange()
        {
            random = new Random((int)DateTime.Now.Ticks);
        }

        /// <summary>
        /// Get a random double between minimum and maximum.
        /// </summary>
        /// <param name="minimum"></param>
        /// <param name="maximum"></param>
        /// <returns></returns>
        public double Range(double minimum, double maximum)
        {
            return random.NextDouble() * (maximum - minimum) + minimum;
        }
    }

}
