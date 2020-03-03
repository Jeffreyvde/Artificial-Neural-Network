﻿using System.Collections.Generic;

namespace NeuralNetwork.Backpropogation
{
    public class GradientDescent
    {
        private readonly List<GradientDecentValue> steps = new List<GradientDecentValue>();

        /// <summary>
        /// Add a value to gradient decent
        /// </summary>
        public void Add(double step, IBackpropogatable backpropogatable)
        {
            steps.Add(new GradientDecentValue(step, backpropogatable));
        }

        /// <summary>
        /// Apply gradient decent 
        /// </summary>
        public void Apply(double learningRate)
        {
            for (int i = 0; i < steps.Count; i++)
            {
                steps[i].Apply(learningRate);
            }
        }

        #region ArithmeticOperators


        /// <summary>
        /// + operator for gradient descents
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static GradientDescent operator +(GradientDescent a, GradientDescent b)
        {
            GradientDescent newDescent = new GradientDescent();
            for (int i = 0; i < a.steps.Count; i++)
            {
                newDescent.Add(a.steps[i].StepValue + b.steps[i].StepValue, a.steps[i].ChangedObject);
            }

            return newDescent;
        }

        /// <summary>
        /// Gradient descent divide
        /// </summary>
        /// <param name="a"></param>
        /// <param name="division"></param>
        /// <returns></returns>
        public static GradientDescent operator /(GradientDescent a, int division)
        {
            for (int i = 0; i < a.steps.Count; i++)
            {
                GradientDecentValue step = a.steps[i];
                step.StepValue /= division;
            }

            return a;
        }


        /// <summary>
        /// Gradient descent *
        /// </summary>
        /// <param name="a"></param>
        /// <param name="times"></param>
        /// <returns></returns>
        public static GradientDescent operator *(GradientDescent a, float times)
        {
            for (int i = 0; i < a.steps.Count; i++)
            {
                GradientDecentValue step = a.steps[i];
                step.StepValue *= times;
            }

            return a;
        }
        #endregion

    }
}
