using System;

namespace NeuralNetwork.Utilities
{
    public static class Random
    {
        public static readonly System.Random random = new System.Random((int)DateTime.Now.Ticks);

        /// <summary>
        /// Get a random double betwen miniumum and maximum.
        /// </summary>
        /// <param name="minimum"></param>
        /// <param name="maximum"></param>
        /// <returns></returns>
        public static double Range(double minimum, double maximum)
        {
            return random.NextDouble() * (maximum - minimum) + minimum;
        }
    }

}
