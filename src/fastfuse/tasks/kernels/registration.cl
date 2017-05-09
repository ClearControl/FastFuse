// #define MAX_GROUP_SIZE  256

__constant sampler_t sampler_nearest = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__constant sampler_t sampler_linear  = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;


__kernel void reduce_mean_1buffer(__global float* dst, __global const float* src) {

  const int gid = get_global_id(0), lid = get_local_id(0);
  __local float sdata[MAX_GROUP_SIZE]; // shared local memory of work group

  const uint group_size = get_local_size(0);
  /*
  if (0==gid && group_size > MAX_GROUP_SIZE) {
    printf("ERROR: local_size too big (%d > %d)\n", group_size, MAX_GROUP_SIZE);
  }
  */ 

  // copy from global to local memory, wait until all work items done
  sdata[lid] = src[gid];
  barrier(CLK_LOCAL_MEM_FENCE);

  // tree-based averaging of all values in the work group
  for(int offset=group_size/2;  offset > 0;  offset /= 2) {
    if (lid < offset) sdata[lid] += sdata[lid + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // write result for this work group
  if (lid == 0) dst[get_group_id(0)] = sdata[0] / group_size;
}


__kernel void reduce_mean_2buffer(__global float* dst1, __global const float* src1, __global float* dst2, __global const float* src2) {

  const int gid = get_global_id(0), lid = get_local_id(0);
  __local float sdata1[MAX_GROUP_SIZE]; // shared local memory of work group
  __local float sdata2[MAX_GROUP_SIZE]; // shared local memory of work group

  const uint group_size = get_local_size(0);
  /*
  if (0==gid && group_size > MAX_GROUP_SIZE) {
    printf("ERROR: local_size too big (%d > %d)\n", group_size, MAX_GROUP_SIZE);
  }
  */ 

  // copy from global to local memory, wait until all work items done
  sdata1[lid] = src1[gid];
  sdata2[lid] = src2[gid];
  barrier(CLK_LOCAL_MEM_FENCE);

  // tree-based averaging of all values in the work group
  for(int offset=group_size/2;  offset > 0;  offset /= 2) {
    if (lid < offset) {
      sdata1[lid] += sdata1[lid + offset];
      sdata2[lid] += sdata2[lid + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // write result for this work group
  if (lid == 0) {
    const int grid = get_group_id(0);
    dst1[grid] = sdata1[0] / group_size;
    dst2[grid] = sdata2[0] / group_size;
  }
}


__kernel void reduce_ncc_affine(__global float* var2, __global float* cov, __read_only image3d_t src1, __read_only image3d_t src2, __constant float* mat, const float m1, const float m2) {

  const uint i = get_global_id(0), j = get_global_id(1), k = get_global_id(2);
  
  const float x = i*mat[0] + j*mat[1] + k*mat[2]  + mat[3];
  const float y = i*mat[4] + j*mat[5] + k*mat[6]  + mat[7];
  const float z = i*mat[8] + j*mat[9] + k*mat[10] + mat[11];

  const float pix1 = read_imagef(src1,sampler_nearest,(int4)(i,j,k,0)).x;
  const float pix2 = read_imagef(src2,sampler_linear,(float4)(0.5f+x,0.5f+y,0.5f+z,0)).x;
  
  const uint4 lsz = (uint4)(get_local_size(0),get_local_size(1),get_local_size(2),0);
  const uint  group_size = lsz.x * lsz.y * lsz.z;
  /*
  if (0==gid && group_size > MAX_GROUP_SIZE) {
    printf("ERROR: lsz too big (%d > %d)\n", group_size, MAX_GROUP_SIZE);
  }
  */ 

  const int lid = get_local_id(0) + lsz.x*get_local_id(1) + lsz.x*lsz.y*get_local_id(2);

  __local float svar2[MAX_GROUP_SIZE]; // shared local memory of work group
  __local float scov[MAX_GROUP_SIZE]; // shared local memory of work group

  // copy from global to local memory, wait until all work items done
  svar2[lid] = (pix2-m2)*(pix2-m2);
  scov[lid]  = (pix1-m1)*(pix2-m2);
  barrier(CLK_LOCAL_MEM_FENCE);

  // tree-based averaging all values in the work group
  for(int offset=group_size/2;  offset > 0;  offset /= 2) {
    if (lid < offset) {
      svar2[lid] += svar2[lid+offset];
      scov[lid]  += scov[lid+offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // write result for this work group
  if (lid == 0) {
    // output size of spatially reduced image (per depth)
    // const uint2 osz = (uint2)(Nx/lsz.x, Ny/lsz.y);
    const uint2 osz = (uint2)(get_num_groups(0),get_num_groups(1));
    const uint  oid = get_group_id(0) + osz.x*get_group_id(1) + (osz.x*osz.y)*get_group_id(2);
    var2[oid] = svar2[0] / group_size;
    cov[oid]  = scov[0]  / group_size;
  }
}


__kernel void affine_transform(__write_only image3d_t dst, __read_only image3d_t src, __constant float* mat) {

  const uint i = get_global_id(0), j = get_global_id(1), k = get_global_id(2);

  const float x = i*mat[0] + j*mat[1] + k*mat[2]  + mat[3];
  const float y = i*mat[4] + j*mat[5] + k*mat[6]  + mat[7];
  const float z = i*mat[8] + j*mat[9] + k*mat[10] + mat[11];

  const float pix = read_imagef(src,sampler_linear,(float4)(0.5f+x,0.5f+y,0.5f+z,0)).x;
  write_imagef(dst,(int4)(i,j,k,0),(float4)(pix,0,0,0));
}


__kernel void reduce_mean_2imagef(__global float* dst1, read_only image3d_t src1, __global float* dst2, read_only image3d_t src2) {

  const uint4 lsz = (uint4)(get_local_size(0),get_local_size(1),get_local_size(2),0);
  const uint  group_size = lsz.x * lsz.y * lsz.z;

  const int4 gid = (int4)(get_global_id(0),get_global_id(1),get_global_id(2),0);
  const int  lid = get_local_id(0) + lsz.x*get_local_id(1) + lsz.x*lsz.y*get_local_id(2);
  /*
  if (0==gid && group_size > MAX_GROUP_SIZE) {
    printf("ERROR: lsz too big (%d > %d)\n", group_size, MAX_GROUP_SIZE);
  }
  */ 

  __local float smean1[MAX_GROUP_SIZE]; // shared local memory of work group
  __local float smean2[MAX_GROUP_SIZE]; // shared local memory of work group

  // copy from global to local memory, wait until all work items done
  smean1[lid] = read_imagef(src1,sampler_nearest,gid).x;
  smean2[lid] = read_imagef(src2,sampler_nearest,gid).x;
  barrier(CLK_LOCAL_MEM_FENCE);

  // tree-based averaging all values in the work group
  for(int offset=group_size/2;  offset > 0;  offset /= 2) {
    if (lid < offset) {
      smean1[lid] += smean1[lid+offset];
      smean2[lid] += smean2[lid+offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // write result for this work group
  if (lid == 0) {
    // output size of spatially reduced image (per depth)
    // const uint2 osz = (uint2)(Nx/lsz.x, Ny/lsz.y);
    const uint2 osz = (uint2)(get_num_groups(0),get_num_groups(1));
    const uint  oid = get_group_id(0) + osz.x*get_group_id(1) + (osz.x*osz.y)*get_group_id(2);
    dst1[oid] = smean1[0] / group_size;
    dst2[oid] = smean2[0] / group_size;
  }
}


__kernel void reduce_var_1imagef(__global float* dst, read_only image3d_t src, const float mean)  {

  const uint4 lsz = (uint4)(get_local_size(0),get_local_size(1),get_local_size(2),0);
  const uint  group_size = lsz.x * lsz.y * lsz.z;

  const int4 gid = (int4)(get_global_id(0),get_global_id(1),get_global_id(2),0);
  const int  lid = get_local_id(0) + lsz.x*get_local_id(1) + lsz.x*lsz.y*get_local_id(2);
  /*
  if (0==gid && group_size > MAX_GROUP_SIZE) {
    printf("ERROR: lsz too big (%d > %d)\n", group_size, MAX_GROUP_SIZE);
  }
  */ 

  __local float svar[MAX_GROUP_SIZE]; // shared local memory of work group

  // copy from global to local memory, wait until all work items done
  const float pix = read_imagef(src,sampler_nearest,gid).x;
  svar[lid] = (pix-mean)*(pix-mean);
  barrier(CLK_LOCAL_MEM_FENCE);

  // tree-based averaging all values in the work group
  for(int offset=group_size/2;  offset > 0;  offset /= 2) {
    if (lid < offset) {
      svar[lid] += svar[lid+offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // write result for this work group
  if (lid == 0) {
    // output size of spatially reduced image (per depth)
    // const uint2 osz = (uint2)(Nx/lsz.x, Ny/lsz.y);
    const uint2 osz = (uint2)(get_num_groups(0),get_num_groups(1));
    const uint  oid = get_group_id(0) + osz.x*get_group_id(1) + (osz.x*osz.y)*get_group_id(2);
    dst[oid] = svar[0] / group_size;
  }
}