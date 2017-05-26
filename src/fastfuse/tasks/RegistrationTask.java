package fastfuse.tasks;

import java.util.concurrent.CopyOnWriteArrayList;

import javax.vecmath.Matrix4f;

import clearcl.ClearCLImage;
import fastfuse.FastFusionEngineInterface;
import fastfuse.registration.AffineMatrix;
import fastfuse.registration.Registration;
import fastfuse.registration.RegistrationParameter;

import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.math3.random.RandomDataGenerator;

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

  // must be a power of 2; number of voxels must be evenly divisible by this
  // suggested: 128 or 256; max value limited by the GPU (e.g., 1024)
  // ideally: number of voxels even divisible by this more than once
  private static final int mGroupSize = 128;

  // number of additional optimization trials with random restarts
  private int mNumberOfRestarts = 4;
  // stop each optimization run after this many function evaluations
  private int mMaxNumberOfEvaluations = 200;
  // voxel scale in z direction (relative to scale 1 for both x and y)
  private float mScaleZ = 4;

  private Matrix4f mZeroTransformMatrix = AffineMatrix.identity();

  // initial transformation (transX, transY, transZ, rotX, rotY, rotZ)
  // rotation angles in degrees around center of volume
  private double[] mInitTransform = new double[]
  { 0, 0, 0, 0, 0, 0 };

  private double mTranslationRadius = 20;
  private double mRotationRadius = 10;

  ///////////////////////////////////////////////////////////////////////////

  private CopyOnWriteArrayList<RegistrationListener> mListenerList =
                                                                   new CopyOnWriteArrayList<>();

  private final RandomDataGenerator mRNG = new RandomDataGenerator();
  private boolean mWaitToFinish = true;
  private String[] mInputImagesSlotKeys;
  private String mTransformedImageSlotKey;
  private Registration mRegistration;

  /**
   * Instantiates a registered fusion task
   * 
   * @param pImageProcessedReferenceSlotKey
   *          first stack (reference volume for registration)
   * @param pImageProcessedToRegisterSlotKey
   *          second stack (volume to be registered to reference volume)
   * @param pImageOriginalReferenceSlotKey
   *          original/reference data for pImageProcessedReferenceSlotKey
   * @param pImageOriginalToRegisterSlotKey
   *          original/reference data for pImageprocessedToRegisterSlotKey, to
   *          be transformed after registration has been found
   * @param pImageOriginalToRegisterTransformedKey
   *          transformed version of pImageOriginalToRegisterSlotKey
   */
  public RegistrationTask(String pImageProcessedReferenceSlotKey,
                          String pImageProcessedToRegisterSlotKey,
                          String pImageOriginalReferenceSlotKey,
                          String pImageOriginalToRegisterSlotKey,
                          String pImageOriginalToRegisterTransformedKey)
  {
    super(pImageProcessedReferenceSlotKey,
          pImageProcessedToRegisterSlotKey,
          pImageOriginalReferenceSlotKey,
          pImageOriginalToRegisterSlotKey);
    mInputImagesSlotKeys = new String[]
    { pImageProcessedReferenceSlotKey,
      pImageProcessedToRegisterSlotKey,
      pImageOriginalReferenceSlotKey,
      pImageOriginalToRegisterSlotKey };
    mTransformedImageSlotKey = pImageOriginalToRegisterTransformedKey;
  }

  @Override
  public boolean enqueue(FastFusionEngineInterface pFastFusionEngine,
                         boolean pWaitToFinish)
  {
    setWaitToFinish(pWaitToFinish);

    ClearCLImage lImageA, lImageB, lImageC, lImageD;
    lImageA = pFastFusionEngine.getImage(mInputImagesSlotKeys[0]);
    lImageB = pFastFusionEngine.getImage(mInputImagesSlotKeys[1]);
    lImageC = pFastFusionEngine.getImage(mInputImagesSlotKeys[2]);
    lImageD = pFastFusionEngine.getImage(mInputImagesSlotKeys[3]);

    if (mRegistration == null)
      mRegistration = new Registration(this, lImageA, lImageB);
    mRegistration.setImages(lImageA, lImageB);

    System.out.println(mRegistration);

    double[] lBestTransform = getInitialTransformation();
    try
    {
      // find best registration
      lBestTransform = mRegistration.register();

      // notify listeners
      notifyListenersOfNewComputedTheta(lBestTransform);

      // and use as initial transform for next time
      setInitialTransformation(lBestTransform);

      mRegistration.setImages(lImageC, lImageD);
      double lBestScoreOriginalImages =
                                      mRegistration.computeScore(lBestTransform);
      System.out.printf("score = %.6f for best transformation on original images\n",
                        lBestScoreOriginalImages);
    }
    catch (Throwable e)
    {
      System.err.println("Finding an updated volume registration failed (using last best transformation parameters instead).\n");
      e.printStackTrace();
    }

    MutablePair<Boolean, ClearCLImage> lFlagAndRegisteredImage =
                                                               pFastFusionEngine.ensureImageAllocated(mTransformedImageSlotKey,
                                                                                                      lImageA.getChannelDataType(),
                                                                                                      lImageA.getDimensions());

    ClearCLImage lRegisteredImage =
                                  lFlagAndRegisteredImage.getRight();
    mRegistration.transform(lRegisteredImage,
                            lImageD,
                            lBestTransform);
    // notify listeners
    notifyListenersOfNewUsedTheta(lBestTransform);

    lFlagAndRegisteredImage.setLeft(true);
    return true;
  }

  /*
   * Interface implementations for fastfuse.registration.RegistrationParameter
   */

  public void setMaxNumberOfEvaluations(int pMaxNumberOfEvaluations)
  {
    assert pMaxNumberOfEvaluations > 0;
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

  public void setScaleZ(float pScaleZ)
  {
    assert pScaleZ > 0;
    mScaleZ = pScaleZ;
  }

  @Override
  public double[] getUpperBounds()
  {
    return new double[]
    { +mTranslationRadius,
      +mTranslationRadius,
      +mTranslationRadius,
      +mRotationRadius,
      +mRotationRadius,
      +mRotationRadius };
  }

  @Override
  public double[] getLowerBounds()
  {
    return new double[]
    { -mTranslationRadius,
      -mTranslationRadius,
      -mTranslationRadius,
      -mRotationRadius,
      -mRotationRadius,
      -mRotationRadius };
  }

  public void setTranslationRadius(double pTranslationRadius)
  {
    mTranslationRadius = pTranslationRadius;
  }

  public void setRotationRadius(double pRotationRadius)
  {
    mRotationRadius = pRotationRadius;
  }

  @Override
  public Matrix4f getZeroTransformMatrix()
  {
    return mZeroTransformMatrix;
  }

  public void setZeroTransformMatrix(Matrix4f pZeroTransformMatrix)
  {
    mZeroTransformMatrix = new Matrix4f(pZeroTransformMatrix);
  }

  @Override
  public double[] getInitialTransformation()
  {
    return mInitTransform;
  }

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

  public void setNumberOfRestarts(int pRestarts)
  {
    assert pRestarts >= 0 && pRestarts < 50;
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
      double c = i < 3 ? mTranslationRadius : mRotationRadius;
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

  private void notifyListenersOfNewComputedTheta(double... theta)
  {
    for (RegistrationListener lRegistrationListener : mListenerList)
      lRegistrationListener.newComputedTheta(theta);
  }

  private void notifyListenersOfNewUsedTheta(double... theta)
  {
    for (RegistrationListener lRegistrationListener : mListenerList)
      lRegistrationListener.newUsedTheta(theta);
  }

  /**
   * Adds a registration listener
   * 
   * @param pRegistrationListener
   *          registration listener
   */
  public void addListener(RegistrationListener pRegistrationListener)
  {
    if (!mListenerList.contains(pRegistrationListener))
      mListenerList.add(pRegistrationListener);
  }

}
