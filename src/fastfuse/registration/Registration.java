package fastfuse.registration;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Stack;
import java.util.stream.LongStream;

import javax.vecmath.Matrix4f;

import clearcl.ClearCLBuffer;
import clearcl.ClearCLContext;
import clearcl.ClearCLImage;
import clearcl.ClearCLKernel;
import clearcl.ClearCLProgram;
import clearcl.enums.ImageChannelDataType;
import clearcl.util.MatrixUtils;
import coremem.enums.NativeTypeEnum;

import org.apache.commons.lang3.reflect.FieldUtils;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.exception.TooManyEvaluationsException;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;

/**
 * Stack registration
 *
 * @author uschmidt
 */
public class Registration
{

  private static final List<String> KERNEL_NAMES =
                                                 Arrays.asList("reduce_mean_1buffer",
                                                               "reduce_mean_2buffer",
                                                               "reduce_mean_3buffer",
                                                               "reduce_mean_2imagef",
                                                               "reduce_var_1imagef",
                                                               "affine_transform",
                                                               "reduce_ncc_affine");

  private final ClearCLContext mContext;
  private final Map<String, ClearCLKernel> mKernels;
  private final RegistrationParameter mParams;

  private long[] mGlobalSize, mLocalSize;
  private final int mGroupSize;
  private final Stack<Long> mBufferSizes = new Stack<>();

  private ClearCLImage mImageA, mImageB;
  private ClearCLBuffer[][] mBuffers;
  private Map<Integer, FloatBuffer> mHostBuffers = new HashMap<>();

  private Matrix4f mMatCenterAndScale, mMatCenterAndScaleInverse;
  private ClearCLBuffer mTransformMatrixBuffer;

  /**
   * Instantiates a stack registration class given two images
   * 
   * @param pParams
   * @param pImageA
   * @param pImageB
   */
  public Registration(RegistrationParameter pParams,
                      ClearCLImage pImageA,
                      ClearCLImage pImageB)
  {
    mParams = pParams;
    mContext = pImageA.getContext();
    mGroupSize = mParams.getOpenCLGroupSize();
    mKernels = getKernels(mGroupSize);
    setSizeAndPrepare(pImageA.getDimensions());
    setImages(pImageA, pImageB);
  }

  public void setImages(ClearCLImage pImageA, ClearCLImage pImageB)
  {
    assert pImageA.getChannelDataType() == ImageChannelDataType.Float;
    assert pImageB.getChannelDataType() == ImageChannelDataType.Float;
    assert Arrays.equals(pImageA.getDimensions(),
                         pImageB.getDimensions());
    mImageA = pImageA;
    mImageB = pImageB;
    if (!hasDimensions(mImageA.getDimensions()))
      setSizeAndPrepare(mImageA.getDimensions());
  }

  private void setSizeAndPrepare(final long... pGlobalSize)
  {
    // set opencl size parameters
    assert 3 == pGlobalSize.length;
    mGlobalSize = pGlobalSize;
    mLocalSize = computeLocalSize(mGroupSize, mGlobalSize);

    // determine buffer sizes
    mBufferSizes.clear();
    mBufferSizes.push(mGlobalSize[0] * mGlobalSize[1]
                      * mGlobalSize[2]);
    mBufferSizes.push(mBufferSizes.peek() / mGroupSize);
    while (mBufferSizes.peek() % mGroupSize == 0
           && mBufferSizes.peek() > mParams.getOpenCLReductionThreshold())
      mBufferSizes.push(mBufferSizes.peek() / mGroupSize);

    // close existing buffers if necessary
    if (mBuffers != null)
    {
      for (int i = 0; i < mBuffers.length; i++)
        for (int j = 0; j < mBuffers[i].length; j++)
          mBuffers[i][j].close();
    }

    // create new buffers
    int lNumReductions = mBufferSizes.size() - 1;
    mBuffers = new ClearCLBuffer[lNumReductions][3];
    for (int i = 0; i < lNumReductions; i++)
      for (int j = 0; j < mBuffers[i].length; j++)
        mBuffers[i][j] =
                       mContext.createBuffer(NativeTypeEnum.Float,
                                             mBufferSizes.get(1 + i));

    // forget host buffers
    mHostBuffers.clear();

    // create fixed matrices for transformations
    long cx = mGlobalSize[0] / 2, cy = mGlobalSize[1] / 2,
        cz = mGlobalSize[2] / 2;
    float sz = mParams.getScaleZ();
    mMatCenterAndScale =
                       AffineMatrix.multiply(AffineMatrix.scaling(1,
                                                                  1,
                                                                  sz),
                                             AffineMatrix.translation(-cx,
                                                                      -cy,
                                                                      -cz));
    mMatCenterAndScaleInverse = new Matrix4f();
    mMatCenterAndScaleInverse.invert(mMatCenterAndScale);
  }

  private static long[] computeLocalSize(final int pGroupSize,
                                         final long... pGlobalSize)
  {
    // sanity checks
    assert pGroupSize > 1;
    final int lGroupSizeExp = (int) (Math.log(pGroupSize)
                                     / Math.log(2));
    assert pGroupSize == Math.pow(2, lGroupSizeExp);
    assert pGlobalSize.length == 3;
    assert (pGlobalSize[0] * pGlobalSize[1] * pGlobalSize[2])
           % pGroupSize == 0;

    // ideal exponents of local sizes
    long[] lExpsIdeal =
    { 0, 0, 0 };
    lExpsIdeal[0] = (long) Math.ceil(lGroupSizeExp / 3.0);
    lExpsIdeal[1] = (long) Math.ceil((lGroupSizeExp - lExpsIdeal[0])
                                     / 2.0);
    lExpsIdeal[2] = lGroupSizeExp - lExpsIdeal[0] - lExpsIdeal[1];
    // corresponding local sizes
    long[] localSize = LongStream.of(lExpsIdeal)
                                 .map(i -> (long) Math.pow(2, i))
                                 .toArray();

    // if (global) image sizes are not multiples of the ideal local sizes
    if ((pGlobalSize[0] % localSize[0] != 0)
        || (pGlobalSize[1] % localSize[1] != 0)
        || (pGlobalSize[2] % localSize[2] != 0))
    {

      List<List<Integer>> lExpsCand = new ArrayList<>();
      for (int i = 0; i < 3; i++)
      {
        lExpsCand.add(new ArrayList<>());
        for (int e = lGroupSizeExp; e >= 0; e--)
          if (pGlobalSize[i] % (int) Math.pow(2, e) == 0)
            lExpsCand.get(i).add(e);
      }

      long[] lExpsChosen = null;
      double minCost = Double.POSITIVE_INFINITY;
      for (int i : lExpsCand.get(0))
        for (int j : lExpsCand.get(1))
          for (int k : lExpsCand.get(2))
            if (lGroupSizeExp == i + j + k)
            {
              double cost = Math.pow(lExpsIdeal[0] - i, 2)
                            + Math.pow(lExpsIdeal[1] - j, 2)
                            + Math.pow(lExpsIdeal[2] - k, 2);
              if (cost < minCost)
              {
                minCost = cost;
                lExpsChosen = new long[]
                { i, j, k };
              }
            }

      localSize =
                LongStream.of(lExpsChosen)
                          .map(i -> (long) Math.pow(2, i))
                          .toArray();
    }
    return localSize;
  }

  public double[] register()
  {
    final float[] meansAB = reduceImageMeans();
    final float varA = reduceImageVar(mImageA, meansAB[0]);
    // System.out.println(Arrays.toString(meansAB));
    // System.out.println(varA);

    MultivariateFunction J = new MultivariateFunction()
    {
      @Override
      public double value(double[] theta)
      {
        float ncc = reduceNCCAffine(floatArray(theta),
                                    meansAB[0],
                                    varA,
                                    meansAB[1]);
        return 1 - ncc;
      }
    };

    // NLopt.NLopt_func Jnlopt = new NLopt.NLopt_func() {
    // @Override
    // public double execute(double[] theta, double[] gradient) {
    // float ncc = reduceNCCAffine(floatArray(theta), meansAB[0], varA,
    // meansAB[1]);
    // return 1 - ncc;
    // }
    // };
    // NLopt optimizer = new NLopt(NLopt.NLOPT_LN_BOBYQA, 6);
    // optimizer.setLowerBounds(mParams.getLowerBounds());
    // optimizer.setUpperBounds(mParams.getUpperBounds());
    // optimizer.setMinObjective(Jnlopt);
    // optimizer.setMaxEval(mParams.getMaxNumberOfEvaluations());

    BOBYQAOptimizer lOptimizer = new BOBYQAOptimizer(2 * 6 + 1);
    SimpleBounds lBounds = new SimpleBounds(mParams.getLowerBounds(),
                                            mParams.getUpperBounds());

    // current best solution is initialization
    double[] initTheta = mParams.getInitialTransformation();
    double[] bestTheta = initTheta;
    double bestJ = J.value(bestTheta);

    // find better registration
    for (int i = 0; i < 1 + mParams.getNumberOfRestarts(); i++)
    {
      // start for optimization
      double[] theta = 0 == i ? initTheta
                              : randomSearch(J, initTheta, 30);
      // mParams.perturbTransformation(initTheta);

      // System.out.printf("init ## %.6f: %s\n", J.value(theta),
      // Arrays.toString(theta));

      // double[] theta_nlopt = theta.clone();
      // NLoptResult minf = optimizer.optimize(theta_nlopt);
      // System.out.printf("nlopt - %.6f: %s\n", minf.minValue(),
      // Arrays.toString(theta_nlopt));

      /////////////////
      try
      {
        lOptimizer.optimize(new MaxEval(mParams.getMaxNumberOfEvaluations()),
                            new ObjectiveFunction(J),
                            GoalType.MINIMIZE,
                            lBounds,
                            new InitialGuess(theta));
      }
      catch (TooManyEvaluationsException e)
      {
      }
      double[] currentTheta = null;
      try
      {
        currentTheta =
                     ((ArrayRealVector) FieldUtils.readField(lOptimizer,
                                                             "currentBest",
                                                             true)).toArray();
      }
      catch (IllegalAccessException e)
      {
        e.printStackTrace();
      }
      /////////////////

      double currentJ = J.value(currentTheta);
      if (currentJ < bestJ)
      {
        bestJ = currentJ;
        bestTheta = currentTheta.clone();
      }

      System.out.printf("run %d - %.6f: %s, iters = %d\n",
                        i + 1,
                        currentJ,
                        Arrays.toString(currentTheta),
                        lOptimizer.getEvaluations());
    }
    System.out.printf("best  = %.6f: %s\n",
                      bestJ,
                      Arrays.toString(bestTheta));
    return bestTheta;
  }

  private double[] randomSearch(MultivariateFunction J,
                                double[] pInitTheta,
                                int pNumberOfSamples)
  {
    double lBestJ = Double.POSITIVE_INFINITY;
    double[] lBestTheta = pInitTheta;
    for (int i = 0; i < pNumberOfSamples; i++)
    {
      // System.out.println(i);
      double[] lTheta = mParams.perturbTransformation(pInitTheta);
      double j = J.value(lTheta);
      if (j < lBestJ)
      {
        lBestTheta = lTheta;
        lBestJ = j;
        // System.out.println(lBestJ);
      }
    }

    return lBestTheta;
  }

  public void transform(ClearCLImage pImageTarget,
                        ClearCLImage pImageSource,
                        double... theta)
  {
    assert pImageSource.getChannelDataType() == ImageChannelDataType.Float;
    assert pImageTarget.getChannelDataType() == ImageChannelDataType.Float;
    ClearCLKernel lKernel = mKernels.get("affine_transform");
    lKernel.setArguments(pImageTarget,
                         pImageSource,
                         getTransformMatrixBuffer(floatArray(theta)));
    lKernel.setGlobalSizes(mGlobalSize);
    lKernel.run(mParams.getWaitToFinish());
  }

  private ClearCLBuffer getTransformMatrixBuffer(float... theta)
  {
    assert theta.length == 6;
    Matrix4f lMatTranslate = AffineMatrix.translation(theta[0],
                                                      theta[1],
                                                      theta[2]);
    Matrix4f lMatRotate = AffineMatrix.rotation(theta[3],
                                                theta[4],
                                                theta[5]);
    Matrix4f lMatFinal =
                       AffineMatrix.multiply(mMatCenterAndScaleInverse,
                                             mParams.getZeroTransformMatrix(),
                                             lMatTranslate,
                                             lMatRotate,
                                             mMatCenterAndScale);
    // lMatFinal.invert();
    mTransformMatrixBuffer = MatrixUtils.matrixToBuffer(mContext,
                                                        mTransformMatrixBuffer,
                                                        lMatFinal);
    return mTransformMatrixBuffer;
  }

  private int checkBuffersAndGetReductionIndex(ClearCLBuffer... bufs)
  {
    assert 1 <= bufs.length && bufs.length <= 3;
    long bufSize = bufs[0].getLength();
    for (int i = 1; i < bufs.length; i++)
      assert bufs[i].getLength() == bufSize;
    for (int i = 0; i < mBufferSizes.size(); i++)
      if (mBufferSizes.get(i) == bufSize)
        return i;
    assert false;
    return -1;
  }

  private float[] reduceMean(ClearCLBuffer... pBuffers)
  {
    int lNumReductions = mBufferSizes.size() - 1;
    int s = checkBuffersAndGetReductionIndex(pBuffers);

    // if no further opencl-based reductions possible
    if (s == lNumReductions)
      return reduceMeanOnHost(pBuffers);

    ClearCLKernel lKernel;
    ClearCLBuffer[] lSrcBuffers;
    switch (pBuffers.length)
    {
    case 1:
      lKernel = mKernels.get("reduce_mean_1buffer");
      lKernel.setLocalSizes(mGroupSize);
      for (int i = s; i < lNumReductions; i++)
      {
        lSrcBuffers = i == s ? pBuffers : mBuffers[i - 1];
        lKernel.setArguments(mBuffers[i][0], lSrcBuffers[0]);
        lKernel.setGlobalSizes(mBufferSizes.get(i));
        lKernel.run(mParams.getWaitToFinish());
      }
      return reduceMeanOnHost(mBuffers[lNumReductions - 1][0]);

    case 2:
      lKernel = mKernels.get("reduce_mean_2buffer");
      lKernel.setLocalSizes(mGroupSize);
      for (int i = s; i < lNumReductions; i++)
      {
        lSrcBuffers = i == s ? pBuffers : mBuffers[i - 1];
        lKernel.setArguments(mBuffers[i][0],
                             lSrcBuffers[0],
                             mBuffers[i][1],
                             lSrcBuffers[1]);
        lKernel.setGlobalSizes(mBufferSizes.get(i));
        lKernel.run(mParams.getWaitToFinish());
      }
      return reduceMeanOnHost(mBuffers[lNumReductions - 1][0],
                              mBuffers[lNumReductions - 1][1]);

    case 3:
      lKernel = mKernels.get("reduce_mean_3buffer");
      lKernel.setLocalSizes(mGroupSize);
      for (int i = s; i < lNumReductions; i++)
      {
        lSrcBuffers = i == s ? pBuffers : mBuffers[i - 1];
        lKernel.setArguments(mBuffers[i][0],
                             lSrcBuffers[0],
                             mBuffers[i][1],
                             lSrcBuffers[1],
                             mBuffers[i][2],
                             lSrcBuffers[2]);
        lKernel.setGlobalSizes(mBufferSizes.get(i));
        lKernel.run(mParams.getWaitToFinish());
      }
      return reduceMeanOnHost(mBuffers[lNumReductions - 1][0],
                              mBuffers[lNumReductions - 1][1],
                              mBuffers[lNumReductions - 1][2]);

    default:
      assert false;
      return null;
    }
  }

  private float[] reduceMeanOnHost(ClearCLBuffer... pBuffers)
  {
    assert pBuffers != null;
    if (pBuffers.length > 1)
    {
      float[] lReductions = new float[pBuffers.length];
      for (int i = 0; i < pBuffers.length; i++)
        lReductions[i] = reduceMeanOnHost(pBuffers[i])[0];
      return lReductions;
    }
    else
    {
      // reuse existing buffer or create new one
      int lBufferSize = (int) pBuffers[0].getLength();
      FloatBuffer lHostBuffer = mHostBuffers.get(lBufferSize);
      if (lHostBuffer == null)
      {
        lHostBuffer = FloatBuffer.allocate(lBufferSize);
        mHostBuffers.put(lBufferSize, lHostBuffer);
      }
      lHostBuffer.clear();
      // transfer to host
      pBuffers[0].writeTo(lHostBuffer, true);
      // naive summation (with double precision)
      double lSum = 0;
      for (int i = 0; i < lBufferSize; i++)
        lSum += lHostBuffer.get(i);
      return new float[]
      { (float) (lSum / lBufferSize) };
    }
  }

  private float[] reduceImageMeans()
  {
    ClearCLKernel lKernel = mKernels.get("reduce_mean_2imagef");
    lKernel.setArguments(mBuffers[0][0],
                         mImageA,
                         mBuffers[0][1],
                         mImageB);
    lKernel.setGlobalSizes(mGlobalSize);
    lKernel.setLocalSizes(mLocalSize);
    lKernel.run(mParams.getWaitToFinish());
    return reduceMean(mBuffers[0][0], mBuffers[0][1]);
  }

  private float reduceImageVar(ClearCLImage pImage, float pMean)
  {
    ClearCLKernel lKernel = mKernels.get("reduce_var_1imagef");
    lKernel.setArguments(mBuffers[0][0], pImage, pMean);
    lKernel.setGlobalSizes(mGlobalSize);
    lKernel.setLocalSizes(mLocalSize);
    lKernel.run(mParams.getWaitToFinish());
    return reduceMean(mBuffers[0][0])[0];
  }

  private float reduceNCCAffine(float[] theta,
                                float meanA,
                                float varA,
                                float meanBapprox)
  {
    // approach:
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data
    ClearCLKernel lKernel = mKernels.get("reduce_ncc_affine");
    lKernel.setArguments(mBuffers[0][0],
                         mBuffers[0][1],
                         mBuffers[0][2],
                         mImageA,
                         mImageB,
                         getTransformMatrixBuffer(theta),
                         meanBapprox);
    lKernel.setGlobalSizes(mGlobalSize);
    lKernel.setLocalSizes(mLocalSize);
    lKernel.run(mParams.getWaitToFinish());
    float[] reds = reduceMean(mBuffers[0][0],
                              mBuffers[0][1],
                              mBuffers[0][2]);
    float meanB = reds[0], meanBB = reds[1], meanAB = reds[2];
    float varB = meanBB - meanB * meanB;
    float covAB = meanAB - meanA * meanB;
    float ncc = (float) (covAB / (Math.sqrt(varA) * Math.sqrt(varB)));
    return ncc;
  }

  private Map<String, ClearCLKernel> getKernels(int pGroupSize)
  {
    HashMap<String, ClearCLKernel> lKernels = null;
    try
    {
      Class<?> lClassForKernel = mParams.getClassForKernelBasePath();
      String lKernelSourceFile = mParams.getKernelSourceFile();
      ClearCLProgram lProgram =
                              mContext.createProgram(lClassForKernel,
                                                     lKernelSourceFile);
      lProgram.addDefine("MAX_GROUP_SIZE", pGroupSize);
      lProgram.addBuildOptionAllMathOpt();
      lProgram.buildAndLog();
      lKernels = new HashMap<>();
      for (String s : KERNEL_NAMES)
        lKernels.put(s, lProgram.createKernel(s));
    }
    catch (IOException e)
    {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
    return lKernels;
  }

  private static float[] floatArray(double[] d)
  {
    assert d != null;
    float[] f = new float[d.length];
    for (int i = 0; i < d.length; i++)
      f[i] = (float) d[i];
    return f;
  }

  private boolean hasDimensions(long... dims)
  {
    if (mGlobalSize == null || dims == null)
      return false;
    return Arrays.equals(mGlobalSize, dims);
  }

  @Override
  public String toString()
  {
    return String.format("Registration(group size = %d):\n"
                         + "-  global size = %4d, %4d, %4d\n"
                         + "-   local size = %4d, %4d, %4d\n"
                         + "- buffer sizes = %s\n",
                         mGroupSize,
                         mGlobalSize[0],
                         mGlobalSize[1],
                         mGlobalSize[2],
                         mLocalSize[0],
                         mLocalSize[1],
                         mLocalSize[2],
                         mBufferSizes.toString());
  }

}
