package fastfuse;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
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
  private final static PrintStream cDebugOut = System.err;

  private static FastFusionMemoryPool mInstance = null;

  private final boolean mDebug;
  private final ClearCLContext mContext;
  private long mPoolSize;
  private long mCurrentSize = 0;
  private final Map<Pair<ImageChannelDataType, List<Long>>, Stack<ClearCLImage>> mImagesAvailable =
                                                                                                  new HashMap<>();
  private final Set<ClearCLImage> mImagesInUse = new HashSet<>();
  private final LinkedHashSet<Pair<ImageChannelDataType, List<Long>>> mImageAccess =
                                                                                   new LinkedHashSet<>();

  public static FastFusionMemoryPool getInstance(ClearCLContext pContext)
  {
    return getInstance(pContext,
                       pContext.getDevice()
                               .getGlobalMemorySizeInBytes(),
                       false);
  }

  public static FastFusionMemoryPool getInstance(ClearCLContext pContext,
                                                 long pPreferredPoolSize)
  {
    if (mInstance == null)
      mInstance = new FastFusionMemoryPool(pContext,
                                           pPreferredPoolSize,
                                           false);
    return mInstance;
  }

  public static FastFusionMemoryPool getInstance(ClearCLContext pContext,
                                                 long pPreferredPoolSize,
                                                 boolean pDebug)
  {
    if (mInstance == null)
      mInstance =
                new FastFusionMemoryPool(pContext,
                                         pPreferredPoolSize,
                                         pDebug);
    return mInstance;
  }

  public static FastFusionMemoryPool get()
  {
    if (mInstance == null)
      throw new FastFusionException("FastFusionMemoryPool.get() can only be called after FastFusionMemoryPool.getInstance()");
    return mInstance;
  }

  private FastFusionMemoryPool(ClearCLContext pContext,
                               long pPoolSize,
                               boolean pDebug)
  {
    mContext = pContext;
    mPoolSize = pPoolSize;
    mDebug = pDebug;
    debug("Creating FastFusionMemoryPool with preferred size limit of %.0f MB\n",
          mPoolSize / (1024d * 1024d));
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
    catch (Throwable e1)
    {
      try
      {
        free(); // free all available images
        lImage =
               mContext.createSingleChannelImage(HostAccessType.ReadWrite,
                                                 KernelAccessType.ReadWrite,
                                                 pDataType,
                                                 pDimensions);
      }
      catch (Throwable e2)
      {
        throw new FastFusionException(e2,
                                      "Couldn't allocate image of type '%s' with dimensions %s",
                                      pDataType.toString(),
                                      Arrays.toString(pDimensions));
      }
    }
    mCurrentSize += lImage.getSizeInBytes();
    return lImage;
  }

  private void freeImage(ClearCLImage pImage)
  {
    mCurrentSize -= pImage.getSizeInBytes();
    pImage.close();
    debug("             free:      %32s - %s\n",
          getKey(pImage).toString(),
          toString());
  }

  private boolean freeMemIsNecessary()
  {
    return mCurrentSize > mPoolSize;
  }

  private boolean freeMemIsPossible()
  {
    return getAvailableImagesCount() > 0;
  }

  private void freeMemIfNecessaryAndPossible()
  {
    if (freeMemIsNecessary() && freeMemIsPossible())
    {
      assert !mImageAccess.isEmpty();
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
    }
  }

  public ClearCLImage requestImage(ImageChannelDataType pDataType,
                                   long... pDimensions)
  {
    return requestImage(null, pDataType, pDimensions);
  }

  public ClearCLImage requestImage(final String pName,
                                   final ImageChannelDataType pDataType,
                                   final long... pDimensions)
  {
    Pair<ImageChannelDataType, List<Long>> lKey = getKey(pDataType,
                                                         pDimensions);
    Stack<ClearCLImage> lSpecificImagesAvailable =
                                                 mImagesAvailable.get(lKey);
    boolean allocated;
    ClearCLImage lImage;
    if (lSpecificImagesAvailable == null
        || lSpecificImagesAvailable.isEmpty())
    {
      allocated = true;
      lImage = allocateImage(pDataType, pDimensions);
    }
    else
    {
      allocated = false;
      lImage = lSpecificImagesAvailable.pop();
    }
    assert !mImagesInUse.contains(lImage);
    mImagesInUse.add(lImage);
    debug("%10s - %s  %32s - %s\n",
          prettyName(pName, 10),
          allocated ? "allocate:" : "reuse:   ",
          lKey.toString(),
          toString());
    if (allocated)
      freeMemIfNecessaryAndPossible();
    return lImage;
  }

  public void releaseImage(ClearCLImage pImage)
  {
    releaseImage(null, pImage);
  }

  public void releaseImage(String pName, ClearCLImage pImage)
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
    debug("%10s - release:   %32s - %s\n",
          prettyName(pName, 10),
          lKey.toString(),
          toString());
    freeMemIfNecessaryAndPossible();
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
    debug("Freeing all available images\n");
    for (Stack<ClearCLImage> lStack : mImagesAvailable.values())
      while (!lStack.isEmpty())
        freeImage(lStack.pop());
    mImagesAvailable.clear();
    mImageAccess.clear();
    if (pFreeImagesInUse)
    {
      debug("Freeing all images that are still in use\n");
      Iterator<ClearCLImage> it = mImagesInUse.iterator();
      while (it.hasNext())
      {
        ClearCLImage lImage = it.next();
        it.remove();
        freeImage(lImage);
      }
      assert mCurrentSize == 0;
    }
  }

  public long getPreferredSizeLimit()
  {
    return mPoolSize;
  }

  public void setPreferredSizeLimit(long pPreferredPoolSize)
  {
    mPoolSize = pPreferredPoolSize;
  }

  public long getCurrentSize()
  {
    return mCurrentSize;
  }

  public boolean isInUse(ClearCLImage pImage)
  {
    return pImage != null && mImagesInUse.contains(pImage);
  }

  @Override
  public String toString()
  {
    return String.format("MemoryPool(used = %2d, avail = %2d, memory = %4.0f | %.0f MB)",
                         mImagesInUse.size(),
                         getAvailableImagesCount(),
                         mCurrentSize / (1024d * 1024d),
                         mPoolSize / (1024d * 1024d));
  }

  private void recordAccess(Pair<ImageChannelDataType, List<Long>> pKey)
  {
    mImageAccess.remove(pKey);
    mImageAccess.add(pKey);
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

  private String prettyName(String pName, int pLength)
  {
    if (pName == null)
      return "<unnamed>";
    if (pName.length() <= pLength)
      return pName;
    return pName.substring(0, pLength - 3) + "...";
  }

  private int getAvailableImagesCount()
  {
    int lNumAvailable = 0;
    for (Stack<ClearCLImage> lStack : mImagesAvailable.values())
      lNumAvailable += lStack.size();
    return lNumAvailable;
  }

  private void debug(String format, Object... args)
  {
    if (mDebug)
      cDebugOut.printf(format, args);
  }

}
