package fastfuse.tasks.regfusion;

import javax.vecmath.Matrix4f;

import clearcl.ClearCLBuffer;
import clearcl.ClearCLImage;
import fastfuse.FastFusionEngineInterface;
import fastfuse.registration.Registration;
import fastfuse.tasks.AverageTask;
import fastfuse.tasks.FusionTaskBase;
import fastfuse.tasks.FusionTaskInterface;

/**
 * Registered fusion task
 *
 * @author royer (@uwe: please add your name to all classes that you touch)
 */
public class RegisteredFusionTask extends FusionTaskBase
                                  implements FusionTaskInterface
{
  private String[] mInputImagesSlotKeys;
  private String mDestImageSlotKey;

  // example:
  private ClearCLBuffer mTempBuffer;

  // class use to do the registration:
  private Registration mRegistration;

  /**
   * Instanciates a registered fusion task
   * 
   * @param pImageASlotKey
   *          first stack
   * @param pImageBSlotKey
   *          second stack
   * @param pDestImageKey
   *          fused stack
   */
  public RegisteredFusionTask(String pImageASlotKey,
                              String pImageBSlotKey,
                              String pDestImageKey)
  {
    super(pImageASlotKey, pImageBSlotKey);
    setupProgram(AverageTask.class, "./kernels/fuseavg.cl"); // replace with
                                                             // another kenel
                                                             // file
    mInputImagesSlotKeys = new String[]
    { pImageASlotKey, pImageBSlotKey };
    mDestImageSlotKey = pDestImageKey;
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

    // if you need a temporary buffer, you want to allocate it lazily and keep
    // it as long as its dimensions are compatible with the images;
    // Let's assume that you ned to copy image A into a temp buffer (would be
    // better if we did not have to... but... sometimes...)
    if (mTempBuffer == null
        || mTempBuffer.getWidth() != lImageA.getWidth()
        || mTempBuffer.getHeight() != lImageA.getHeight()
        || mTempBuffer.getDepth() != lImageA.getDepth())
    {
      if (mTempBuffer != null)
        mTempBuffer.close(); // we need to make sure that we dispose of old
                             // buffers with teh wrong size
      lImageA.copyTo(mTempBuffer, false); // no need to block, as more can eb
                                          // enqueued.
    }

    // Of course you have to first register the two stacks before doing the
    // fusion.
    // Ideally, that code would be put in a separate class to avoid making this
    // class too long,
    // that 'registration' class would take some parameters.
    // keep in mind that the registration class can be used to hold 'state'
    // information such as the
    // last found registration transform - this is usefull so that successive
    // registrations are
    // warm starts...

    mRegistration.register(); // add some parameters here

    // after that you have to enqueue kernels for fusion ...

    // How do we get the image that we write to?
    // Just use the slot key defined for that image to retreive an imageof right
    // dimensions, and data type:
    ClearCLImage lFusedImage = pFastFusionEngine
                                                .ensureImageAllocated(mDestImageSlotKey,
                                                                      lImageA.getChannelDataType(),
                                                                      lImageA.getDimensions())
                                                .getRight();

    // voila! hope it helps!

    return true;
  }

}
