package fastfuse.tasks;

import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.math3.random.RandomDataGenerator;

import clearcl.ClearCLImage;
import fastfuse.FastFusionEngineInterface;
import fastfuse.registration.Registration;
import fastfuse.registration.RegistrationParameter;

/**
 * Stack registration. This task takes two images, and applies an affine
 * transform to the second in order to register it to the first. The output is
 * the transformed (second) image.
 *
 * @author royer, uschmidt
 */
public class RegistrationTask extends TaskBase
		implements TaskInterface, RegistrationParameter {

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

	// initial transformation (transX, transY, transZ, rotX, rotY, rotZ)
	// rotation angles in degrees around center of volume
	private double[] mInitTransform = new double[] { 0, 0, 0, 0, 0, 0 };
	// bounds for transformation parameters
	// private static final double[] mLowerBnd = new double[] { -30, -30, -30,
	// -10, -10, -10 };
	// private static final double[] mUpperBnd = new double[] { +30, +30, +30,
	// +10, +10, +10 };
	// private double[] mInitTransform = new double[] { 0, 0, 0, 0, 180, 0 };
	// private static final double[] mLowerBnd = new double[] { -30, -30, -30,
	// -10, 170, -10 };
	// private static final double[] mUpperBnd = new double[] { +30, +30, +30,
	// +10, 190, +10 };
	//
	private static final double PINF = Double.POSITIVE_INFINITY;
	private static final double NINF = Double.NEGATIVE_INFINITY;
	private double[] mLowerBnd = new double[] { NINF, NINF, NINF, NINF, NINF,
			NINF };
	private double[] mUpperBnd = new double[] { PINF, PINF, PINF, PINF, PINF,
			PINF };

	// random perturbation offsets (+,-) for translation and rotation
	private static final int mRandRotPM = 5;
	private static final int mRandTransPM = 30;

	///////////////////////////////////////////////////////////////////////////

	private final RandomDataGenerator mRNG = new RandomDataGenerator();
	private String[] mInputImagesSlotKeys;
	private String mTransformedImageSlotKey;
	private Registration mRegistration;

	/**
	 * Instantiates a registered fusion task
	 * 
	 * @param pImageASlotKey
	 *            first stack
	 * @param pImageBSlotKey
	 *            second stack
	 * @param pImageBTransformedKey
	 *            transformed stack
	 */
	public RegistrationTask(String pImageASlotKey, String pImageBSlotKey,
			String pImageBTransformedKey) {
		super(pImageASlotKey, pImageBSlotKey);
		mInputImagesSlotKeys = new String[] { pImageASlotKey, pImageBSlotKey };
		mTransformedImageSlotKey = pImageBTransformedKey;
	}

	@Override
	public boolean enqueue(FastFusionEngineInterface pFastFusionEngine,
			boolean pWaitToFinish) {
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

		MutablePair<Boolean, ClearCLImage> lFlagAndRegisteredImage = pFastFusionEngine
				.ensureImageAllocated(mTransformedImageSlotKey,
						lImageA.getChannelDataType(), lImageA.getDimensions());

		ClearCLImage lRegisteredImage = lFlagAndRegisteredImage.getRight();
		mRegistration.transform(lRegisteredImage, lImageB, lBestTransform);
		lFlagAndRegisteredImage.setLeft(true);
		return true;
	}

	/*
	 * Interface implementations for fastfuse.registration.RegistrationParameter
	 */

	@Override
	public void setMaxNumberOfEvaluations(int pMaxNumberOfEvaluations) {
		mMaxNumberOfEvaluations = pMaxNumberOfEvaluations;
	}

	@Override
	public int getMaxNumberOfEvaluations() {
		return mMaxNumberOfEvaluations;
	}

	@Override
	public float getScaleZ() {
		return mScaleZ;
	}

	@Override
	public void setScaleZ(float pScaleZ) {
		mScaleZ = pScaleZ;
	}

	@Override
	public double[] getUpperBounds() {
		return mUpperBnd;
	}

	@Override
	public double[] getLowerBounds() {
		return mLowerBnd;
	}

	@Override
	public void setLowerBounds(double[] pLowerBound) {
		assert 6 == pLowerBound.length;
		mLowerBnd = pLowerBound;
	}

	@Override
	public void setUpperBounds(double[] pUpperBound) {
		assert 6 == pUpperBound.length;
		mUpperBnd = pUpperBound;
	}

	@Override
	public double[] getInitialTransformation() {
		return mInitTransform;
	}

	@Override
	public void setInitialTransformation(double... theta) {
		assert theta.length == 6;
		mInitTransform = theta;
	}

	@Override
	public Class<?> getClassForKernelBasePath() {
		return this.getClass();
	}

	@Override
	public String getKernelSourceFile() {
		return "./kernels/registration.cl";
	}

	@Override
	public int getNumberOfRestarts() {
		return mNumberOfRestarts;
	}

	@Override
	public void setNumberOfRestarts(int pRestarts) {
		assert pRestarts > 0 && pRestarts < 50;
		mNumberOfRestarts = pRestarts;
	}

	@Override
	public int getOpenCLGroupSize() {
		return mGroupSize;
	}

	@Override
	public double[] perturbTransformation(double... theta) {
		assert theta.length == 6;
		double[] perturbed = new double[theta.length];
		double[] lb = getLowerBounds(), ub = getUpperBounds();
		for (int i = 0; i < theta.length; i++) {
			int c = i < 3 ? mRandTransPM : mRandRotPM;
			perturbed[i] = theta[i] + mRNG.nextUniform(-c, c);
			perturbed[i] = Math.max(lb[i], perturbed[i]);
			perturbed[i] = Math.min(ub[i], perturbed[i]);
		}
		return perturbed;
	}

	@Override
	public boolean getWaitToFinish() {
		return mWaitToFinish;
	}

	public void setWaitToFinish(boolean pWaitToFinish) {
		mWaitToFinish = pWaitToFinish;
	}

}
