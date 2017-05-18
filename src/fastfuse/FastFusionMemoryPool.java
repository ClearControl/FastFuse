package fastfuse;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

import clearcl.ClearCLContext;
import clearcl.ClearCLImage;
import clearcl.enums.HostAccessType;
import clearcl.enums.ImageChannelDataType;
import clearcl.enums.KernelAccessType;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.util.Pair;

public class FastFusionMemoryPool implements AutoCloseable
{

  private static FastFusionMemoryPool mInstance = null;

  private final ClearCLContext mContext;
  private final long mPoolSize;
  private long mCurrentSize = 0;
  private final Map<Pair<ImageChannelDataType, List<Long>>, Stack<ClearCLImage>> mImagesAvailable =
                                                                                                  new HashMap<>();
  private final Set<ClearCLImage> mImagesInUse = new HashSet<>();
  private final LinkedHashSet<Pair<ImageChannelDataType, List<Long>>> mImageAccess =
                                                                                   new LinkedHashSet<>();

  public static FastFusionMemoryPool get(ClearCLContext pContext)
  {
    return get(pContext, Long.MAX_VALUE);
  }

  public static FastFusionMemoryPool get(ClearCLContext pContext,
                                         long pPreferredPoolSize)
  {
    if (mInstance == null)
      mInstance = new FastFusionMemoryPool(pContext,
                                           pPreferredPoolSize);
    return mInstance;
  }

  private FastFusionMemoryPool(ClearCLContext pContext,
                               long pPoolSize)
  {
    mContext = pContext;
    mPoolSize = pPoolSize;
  }

  private ClearCLImage allocateImage(ImageChannelDataType pDataType,
                                     long... pDimensions)
  {
    ClearCLImage lImage;
    try
    {
      lImage =
             mContext.createSingleChannelImage(HostAccessType.ReadWrite,
                                               KernelAccessType.ReadWrite,
                                               pDataType,
                                               pDimensions);
    }
    catch (Exception e1)
    {
      try
      {
        freeMem();
        lImage =
               mContext.createSingleChannelImage(HostAccessType.ReadWrite,
                                                 KernelAccessType.ReadWrite,
                                                 pDataType,
                                                 pDimensions);
      }
      catch (Exception e2)
      {
        e1.printStackTrace();
        e2.printStackTrace();
        return null;
      }
    }
    mCurrentSize += lImage.getSizeInBytes();
    return lImage;
  }

  private void freeImage(ClearCLImage pImage)
  {
    mCurrentSize -= pImage.getSizeInBytes();
    pImage.close();
  }

  private boolean freeMemIsNecessary()
  {
    return mCurrentSize > mPoolSize;
  }

  private boolean freeMemIsPossible()
  {
    return getAvailableImagesCount() > 0;
  }

  private void freeMem()
  {
    if (freeMemIsNecessary() && freeMemIsPossible())
    {
      assert !mImageAccess.isEmpty();
      // long lSizeBefore = mCurrentSize;
      for (Pair<ImageChannelDataType, List<Long>> lKeyAccess : mImageAccess)
      {
        Stack<ClearCLImage> lImageStack =
                                        mImagesAvailable.get(lKeyAccess);
        assert lImageStack != null;
        while (!lImageStack.isEmpty() && freeMemIsNecessary())
        {
          freeImage(lImageStack.pop());
        }
      }
      // System.err.printf("Freeing %.1f MB of memory - ", (lSizeBefore -
      // mCurrentSize)/(1024d*1024d));
    }
  }

  public ClearCLImage requestImage(final ImageChannelDataType pDataType,
                                   final long... pDimensions)
  {
    Pair<ImageChannelDataType, List<Long>> lKey = getKey(pDataType,
                                                         pDimensions);
    Stack<ClearCLImage> lSpecificImagesAvailable =
                                                 mImagesAvailable.get(lKey);
    ClearCLImage lImage;
    if (lSpecificImagesAvailable == null
        || lSpecificImagesAvailable.isEmpty())
    {
      lImage = allocateImage(pDataType, pDimensions);
      freeMem();
    }
    else
    {
      lImage = lSpecificImagesAvailable.pop();
    }
    assert !mImagesInUse.contains(lImage);
    mImagesInUse.add(lImage);
    // System.err.printf("REQuest: %32s - %s\n", lKey.toString(), debug());
    return lImage;
  }

  public void releaseImage(ClearCLImage pImage)
  {
    assert mImagesInUse.contains(pImage);
    mImagesInUse.remove(pImage);
    Pair<ImageChannelDataType, List<Long>> lKey = getKey(pImage);
    recordAccess(lKey);
    Stack<ClearCLImage> lSpecificImagesAvailable =
                                                 mImagesAvailable.get(lKey);
    if (lSpecificImagesAvailable == null)
    {
      lSpecificImagesAvailable = new Stack<>();
      mImagesAvailable.put(lKey, lSpecificImagesAvailable);
    }
    lSpecificImagesAvailable.push(pImage);
    freeMem();
    // System.err.printf("RELease: %32s - %s\n", getKey(pImage).toString(),
    // debug());
  }

  private void recordAccess(Pair<ImageChannelDataType, List<Long>> pKey)
  {
    mImageAccess.remove(pKey);
    mImageAccess.add(pKey);
  }

  public boolean isInUse(ClearCLImage pImage)
  {
    return pImage != null && mImagesInUse.contains(pImage);
  }

  @Override
  public void close() throws Exception
  {
    free(true);
  }

  public void free()
  {
    free(false);
  }

  public void free(boolean pFreeImagesInUse)
  {
    for (Stack<ClearCLImage> lStack : mImagesAvailable.values())
    {
      for (ClearCLImage lImage : lStack)
        freeImage(lImage);
      lStack.clear();
    }
    mImagesAvailable.clear();
    mImageAccess.clear();
    if (pFreeImagesInUse)
    {
      for (ClearCLImage lImage : mImagesInUse)
        freeImage(lImage);
      mImagesInUse.clear();
    }
    assert mCurrentSize == 0;
  }

  private Pair<ImageChannelDataType, List<Long>> getKey(final ClearCLImage pImage)
  {
    return getKey(pImage.getChannelDataType(),
                  pImage.getDimensions());
  }

  private Pair<ImageChannelDataType, List<Long>> getKey(final ImageChannelDataType pDataType,
                                                        final long... pDimensions)
  {
    return Pair.create(pDataType,
                       Arrays.asList(ArrayUtils.toObject(pDimensions)));
  }

  @Override
  public String toString()
  {
    return debug()
           + String.format(", current size = %.1f MB, preferred size limit = %.1f MB",
                           mCurrentSize / (1024d * 1024d),
                           mPoolSize / (1024d * 1024d));
  }

  private String debug()
  {
    return String.format("MemoryPool(in use = %2d, available = %2d)",
                         mImagesInUse.size(),
                         getAvailableImagesCount());
  }

  private int getAvailableImagesCount()
  {
    int lNumAvailable = 0;
    for (Stack<ClearCLImage> lStack : mImagesAvailable.values())
      lNumAvailable += lStack.size();
    return lNumAvailable;
  }

}
