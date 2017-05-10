package fastfuse.tasks;

import javax.vecmath.Matrix4f;

import clearcl.ClearCLImage;
import fastfuse.FastFusionEngineInterface;
import fastfuse.registration.Registration;
import fastfuse.registration.RegistrationParameter;

import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.math3.random.RandomDataGenerator;
import simbryo.util.geom.GeometryUtils;

/**
 * Stack registration. This task takes two images, and applies an affine
 * transform to the second in order to register it to the first. The output is
 * the transformed (second) image.
 *
 * @author royer, uschmidt
 */
public class RegistrationTask extends TaskBase implements
                              TaskInterface,
                              RegistrationParameter
{

  ///////////////////////////////////////////////////////////////////////////
  // (DEFAULT) PARAMETERS
  ///////////////////////////////////////////////////////////////////////////

  // at most 256 (or what's specified in the opencl source file)
  // must be a power of 2
  private static final int mGroupSize = 128;

  // wait for opencl kernels to finish
  private boolean mWaitToFinish = false;

  // number of optimization trials with random restarts
  private int mNumberOfRestarts = 5;
  // stop each optimization run after this many function evaluations
  private int mMaxNumberOfEvaluations = 200;
  // voxel scale in z direction (relative to scale 1 for both x and y)
  private float mScaleZ = 4;

  private Matrix4f mZeroTransformMatrix = GeometryUtils.getIdentity();

  // initial transformation (transX, transY, transZ, rotX, rotY, rotZ)
  // rotation angles in degrees around center of volume
  private double[] mInitTransform = new double[]
  { 0, 0, 0, 0, 0, 0 };

  private static final double cTranslationRadius = 20;
  private static final double cRotationRadius = 10;
  private double[] mLowerBnd = new double[]
  { -cTranslationRadius,
    -cTranslationRadius,
    -cTranslationRadius,
    -cRotationRadius,
    -cRotationRadius,
    -cRotationRadius };
  private double[] mUpperBnd = new double[]
  { +cTranslationRadius,
    +cTranslationRadius,
    +cTranslationRadius,
    cRotationRadius,
    cRotationRadius,
    cRotationRadius };

  ///////////////////////////////////////////////////////////////////////////

  private final RandomDataGenerator mRNG = new RandomDataGenerator();
  private String[] mInputImagesSlotKeys;
  private String mTransformedImageSlotKey;
  private Registration mRegistration;

  /**
   * Instantiates a registered fusion task
   * 
   * @param pImageASlotKey
   *          first stack
   * @param pImageBSlotKey
   *          second stack
   * @param pImageBTransformedKey
   *          transformed stack
   */
  public RegistrationTask(String pImageASlotKey,
                          String pImageBSlotKey,
                          String pImageBTransformedKey)
  {
    super(pImageASlotKey, pImageBSlotKey);
    mInputImagesSlotKeys = new String[]
    { pImageASlotKey, pImageBSlotKey };
    mTransformedImageSlotKey = pImageBTransformedKey;
  }

  @Override
  public boolean enqueue(FastFusionEngineInterface pFastFusionEngine,
                         boolean pWaitToFinish)
  {
    setWaitToFinish(pWaitToFinish);

    ClearCLImage lImageA, lImageB;
    lImageA = pFastFusionEngine.getImage(mInputImagesSlotKeys[0]);
    lImageB = pFastFusionEngine.getImage(mInputImagesSlotKeys[1]);

    if (mRegistration == null)
      mRegistration = new Registration(this, lImageA, lImageB);
    mRegistration.setImages(lImageA, lImageB);

    System.out.println(mRegistration);

    // find best registration
    double[] lBestTransform = mRegistration.register();
    // and use as initial transform for next time
    setInitialTransformation(lBestTransform);

    MutablePair<Boolean, ClearCLImage> lFlagAndRegisteredImage =
                                                               pFastFusionEngine.ensureImageAllocated(mTransformedImageSlotKey,
                                                                                                      lImageA.getChannelDataType(),
                                                                                                      lImageA.getDimensions());

    ClearCLImage lRegisteredImage =
                                  lFlagAndRegisteredImage.getRight();
    mRegistration.transform(lRegisteredImage,
                            lImageB,
                            lBestTransform);
    lFlagAndRegisteredImage.setLeft(true);
    return true;
  }

  /*
   * Interface implementations for fastfuse.registration.RegistrationParameter
   */

  @Override
  public void setMaxNumberOfEvaluations(int pMaxNumberOfEvaluations)
  {
    mMaxNumberOfEvaluations = pMaxNumberOfEvaluations;
  }

  @Override
  public int getMaxNumberOfEvaluations()
  {
    return mMaxNumberOfEvaluations;
  }

  @Override
  public float getScaleZ()
  {
    return mScaleZ;
  }

  @Override
  public void setScaleZ(float pScaleZ)
  {
    mScaleZ = pScaleZ;
  }

  @Override
  public double[] getUpperBounds()
  {
    return mUpperBnd;
  }

  @Override
  public double[] getLowerBounds()
  {
    return mLowerBnd;
  }

  @Override
  public void setLowerBounds(double[] pLowerBound)
  {
    assert 6 == pLowerBound.length;
    mLowerBnd = pLowerBound;
  }

  @Override
  public void setUpperBounds(double[] pUpperBound)
  {
    assert 6 == pUpperBound.length;
    mUpperBnd = pUpperBound;
  }

  @Override
  public Matrix4f getZeroTransformMatrix()
  {
    return mZeroTransformMatrix;
  }

  @Override
  public void setZeroTransformMatrix(Matrix4f pZeroTransformMatrix)
  {
    mZeroTransformMatrix = new Matrix4f(pZeroTransformMatrix);
  }

  @Override
  public double[] getInitialTransformation()
  {
    return mInitTransform;
  }

  @Override
  public void setInitialTransformation(double... theta)
  {
    assert theta.length == 6;
    mInitTransform = theta;
  }

  @Override
  public Class<?> getClassForKernelBasePath()
  {
    return this.getClass();
  }

  @Override
  public String getKernelSourceFile()
  {
    return "./kernels/registration.cl";
  }

  @Override
  public int getNumberOfRestarts()
  {
    return mNumberOfRestarts;
  }

  @Override
  public void setNumberOfRestarts(int pRestarts)
  {
    assert pRestarts > 0 && pRestarts < 50;
    mNumberOfRestarts = pRestarts;
  }

  @Override
  public int getOpenCLGroupSize()
  {
    return mGroupSize;
  }

  @Override
  public double[] perturbTransformation(double... theta)
  {
    assert theta.length == 6;
    double[] lPerturbedTheta = new double[theta.length];
    double[] lb = getLowerBounds(), ub = getUpperBounds();
    for (int i = 0; i < theta.length; i++)
    {
      double c = i < 3 ? cTranslationRadius : cRotationRadius;
      lPerturbedTheta[i] = theta[i] + mRNG.nextUniform(-c, c);
      lPerturbedTheta[i] = Math.max(lb[i], lPerturbedTheta[i]);
      lPerturbedTheta[i] = Math.min(ub[i], lPerturbedTheta[i]);
    }
    return lPerturbedTheta;
  }

  @Override
  public boolean getWaitToFinish()
  {
    return mWaitToFinish;
  }

  public void setWaitToFinish(boolean pWaitToFinish)
  {
    mWaitToFinish = pWaitToFinish;
  }

}
