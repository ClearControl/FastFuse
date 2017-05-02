package fastfuse.tasks;

import javax.vecmath.Matrix4f;

import clearcl.ClearCLBuffer;
import clearcl.ClearCLImage;
import fastfuse.FastFusionEngineInterface;
import fastfuse.registration.Registration;

/**
 * Stack registration. This task takes two images, and applies a rigid transform
 * to the second in order to register it to the first. The output is the
 * transformed (second) image.
 *
 * @author royer (@uwe: please add your name to all classes that you touch)
 */
public class RegistrationTask extends TaskBase
                              implements TaskInterface
{
  private String[] mInputImagesSlotKeys;
  private String mTransformedImageSlotKey;

  // example:
  private ClearCLBuffer mTempBuffer;

  // class use to do the registration:
  private Registration mRegistration = new Registration();

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
    setupProgram(AverageTask.class, "./kernels/fuseavg.cl"); // replace with
    // another
    // source
    // file
    // file
    mInputImagesSlotKeys = new String[]
    { pImageASlotKey, pImageBSlotKey };
    mTransformedImageSlotKey = pImageBTransformedKey;
  }

  /**
   * Sets the initial matrix used for registration
   * 
   * @param pInitialMatrix
   *          initial matrix
   */
  public void setInitialTransformMatrix(Matrix4f pInitialMatrix)
  {
    mRegistration.setInitialTransformMatrix(pInitialMatrix);
  }

  /**
   * Sets max number of iterations
   * 
   * @param pMaxNumberOfIterations
   *          maximal number of iterations
   */
  public void setMaxNumberOfIterations(int pMaxNumberOfIterations)
  {
    mRegistration.setMaxNumberOfIterations(pMaxNumberOfIterations);
  }

  @Override
  public boolean enqueue(FastFusionEngineInterface pFastFusionEngine,
                         boolean pWaitToFinish)
  {
    // Typically, you first get a hold on the images:
    ClearCLImage lImageA, lImageB;

    lImageA = pFastFusionEngine.getImage(mInputImagesSlotKeys[0]);
    lImageB = pFastFusionEngine.getImage(mInputImagesSlotKeys[1]);

    // if you need a temporary buffer, you want to allocate it lazily and
    // keep
    // it as long as its dimensions are compatible with the images;
    // Let's assume that you ned to copy image A into a temp buffer (would
    // be
    // better if we did not have to... but... sometimes...)
    if (mTempBuffer == null
        || mTempBuffer.getWidth() != lImageA.getWidth()
        || mTempBuffer.getHeight() != lImageA.getHeight()
        || mTempBuffer.getDepth() != lImageA.getDepth())
    {
      if (mTempBuffer != null)
        mTempBuffer.close(); // we need to make sure that we dispose of
      // old
      // buffers with teh wrong size
      lImageA.copyTo(mTempBuffer, false); // no need to block, as more can
      // eb
      // enqueued.
    }

    // Of course you have to first register the two stacks before doing the
    // fusion.
    // Ideally, that code would be put in a separate class to avoid making
    // this
    // class too long,
    // that 'registration' class would take some parameters.
    // keep in mind that the registration class can be used to hold 'state'
    // information such as the
    // last found registration transform - this is usefull so that
    // successive
    // registrations are
    // warm starts...

    mRegistration.register(); // add some parameters here

    // after that you have to enqueue kernels for fusion ...

    // How do we get the image that we write to?
    // Just use the slot key defined for that image to retreive an imageof
    // right
    // dimensions, and data type:
    ClearCLImage lFusedImage = pFastFusionEngine
                                                .ensureImageAllocated(mTransformedImageSlotKey,
                                                                      lImageA.getChannelDataType(),
                                                                      lImageA.getDimensions())
                                                .getRight();

    // voila! hope it helps!

    return true;
  }

}
