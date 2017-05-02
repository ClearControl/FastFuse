package fastfuse.tasks;

import fastfuse.FastFusionEngineInterface;

/**
 * Identity task - this task does nothing, just instantaneously passes the image
 * from a source to a destination slot.
 *
 * @author royer
 */
public class IdentityTask extends TaskBase
                          implements TaskInterface
{

  private final String mSrcImageSlotKey, mDstImageSlotKey;

  /**
   * Instantiates an identity task.
   * 
   * @param pSrcImageSlotKey
   *          source slot key
   * @param pDstImageSlotKey
   *          destination slot key
   */
  public IdentityTask(String pSrcImageSlotKey,
                      String pDstImageSlotKey)
  {
    mSrcImageSlotKey = pSrcImageSlotKey;
    mDstImageSlotKey = pDstImageSlotKey;
  }

  @Override
  public boolean enqueue(FastFusionEngineInterface pFastFusionEngine,
                         boolean pWaitToFinish)
  {
    pFastFusionEngine.assignImageToAnotherSlotKey(mSrcImageSlotKey,
                                                  mDstImageSlotKey);
    return true;
  }

}
