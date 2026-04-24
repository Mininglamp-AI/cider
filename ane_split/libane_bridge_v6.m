// libane_bridge_v6.m — ANE bridge with manual retain (no ARC), IOSurface isolation
// Compile: clang -shared -o libane_bridge_v6.dylib libane_bridge_v6.m \
//   -framework Foundation -framework IOSurface -framework Accelerate -lobjc -O2 -fno-objc-arc
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <dispatch/dispatch.h>

static Class g_D, g_I, g_AR, g_AIO;
static int g_init = 0;
static dispatch_queue_t g_ane_queue;
static dispatch_group_t g_ane_group;

typedef struct {
    id model;      // retained manually
    int ic, oc, seq;
    int ioInIdx, ioOutIdx;
} ANEModel;

#define MAX_MODELS 256
#define MAX_SURFACES 32
static ANEModel g_models[MAX_MODELS];
static id g_requests[MAX_MODELS];  // retained manually
static int g_count = 0;

typedef struct {
    IOSurfaceRef surface;
    id aneObj;     // retained manually
    size_t allocSize;
} IOSEntry;
static IOSEntry g_surfaces[MAX_SURFACES];
static int g_surfCount = 0;

static int find_or_create_surface(size_t needed, int exclude_idx) {
    needed = (needed + 0xFFFF) & ~(size_t)0xFFFF;
    if (needed < 65536) needed = 65536;
    for (int i = 0; i < g_surfCount; i++) {
        if (g_surfaces[i].allocSize == needed && i != exclude_idx) return i;
    }
    if (g_surfCount >= MAX_SURFACES) return -1;
    IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(needed),(id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(needed),
        (id)kIOSurfaceAllocSize:@(needed),(id)kIOSurfacePixelFormat:@0});
    if (!s) return -1;
    id obj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO,@selector(objectWithIOSurface:),s);
    [obj retain];
    int idx = g_surfCount++;
    g_surfaces[idx] = (IOSEntry){s, obj, needed};
    return idx;
}

static NSData *build_blob(const float *w, int oc, int ic) {
    size_t n = (size_t)oc * ic, wsize = n * 2, total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE; chunk[4]=0x01;
    *(uint32_t*)(chunk+8) = (uint32_t)wsize;
    *(uint32_t*)(chunk+16) = 128;
    _Float16 *fp16 = (_Float16*)(buf + 128);
    for (size_t i = 0; i < n; i++) fp16[i] = (_Float16)w[i];
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSString *gen_conv_mil(int ic, int oc, int seq) {
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [%d, %d, 1, 1]>"
        "(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, "
        "pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n}\n",
        ic,seq, ic,seq, oc,ic,oc,ic, oc,seq, oc,seq];
}

// INT8 MIL text: uses constexpr_blockwise_shift_scale for INT8 weight quantization
static NSString *gen_conv_mil_int8(int ic, int oc, int seq) {
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\
{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        tensor<int8, [%d, %d, 1, 1]> W_data = const()[name = string(\"W_data\"), "
        "val = tensor<int8, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight_data.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [%d, 1, 1, 1]> W_scale = const()[name = string(\"W_scale\"), "
        "val = tensor<fp16, [%d, 1, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight_scale.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [%d, 1, 1, 1]> W_offset = const()[name = string(\"W_offset\"), "
        "val = tensor<fp16, [%d, 1, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight_offset.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_blockwise_shift_scale("
        "data = W_data, scale = W_scale, offset = W_offset)[name = string(\"dequant\")];\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, "
        "pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n}\n",
        ic,seq, ic,seq,
        oc,ic,oc,ic,    // W_data shape
        oc,oc,          // W_scale shape
        oc,oc,          // W_offset shape
        oc,ic,          // W dequantized shape
        oc,seq,         // y16 shape
        oc,seq];        // y shape
}

// Build INT8 weight blob: header(64) + magic(64) + data
static NSData *build_blob_int8(const int8_t *data, int count) {
    size_t wsize = (size_t)count, total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE; chunk[4]=0x01;
    *(uint32_t*)(chunk+8) = (uint32_t)wsize;
    *(uint32_t*)(chunk+16) = 128;
    memcpy(buf + 128, data, wsize);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSData *build_blob_fp16(const _Float16 *data, int count) {
    size_t wsize = (size_t)count * 2, total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE; chunk[4]=0x01;
    *(uint32_t*)(chunk+8) = (uint32_t)wsize;
    *(uint32_t*)(chunk+16) = 128;
    memcpy(buf + 128, data, wsize);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static int setup_model(id mdl, int ic, int oc, int seq) {
    size_t inB = (size_t)ic * seq * 4;
    size_t outB = (size_t)oc * seq * 4;
    int ioInIdx = find_or_create_surface(inB, -1);
    int ioOutIdx = find_or_create_surface(outB, ioInIdx); // exclude input surface
    if (ioInIdx < 0 || ioOutIdx < 0) return -1;

    id wI = g_surfaces[ioInIdx].aneObj;
    id wO = g_surfaces[ioOutIdx].aneObj;
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    [req retain];
    [mdl retain];

    int handle = g_count;
    g_models[handle] = (ANEModel){mdl, ic, oc, seq, ioInIdx, ioOutIdx};
    g_requests[handle] = req;
    g_count++;
    return handle;
}

// ===== Public C API =====

int ane_init(void) {
    if (g_init) return 0;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!g_D || !g_I || !g_AR || !g_AIO) return -1;
    g_ane_queue = dispatch_queue_create("ane.eval", DISPATCH_QUEUE_SERIAL);
    g_ane_group = dispatch_group_create();
    g_init = 1;
    return 0;
}

int ane_load_model(int ic, int oc, int seq, const float *w) {
    if (!g_init || g_count >= MAX_MODELS) return -1;
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

    NSData *mil = [gen_conv_mil(ic, oc, seq) dataUsingEncoding:NSUTF8StringEncoding];
    NSData *blob = build_blob(w, oc, ic);
    NSDictionary *wd = @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":blob}};
    NSError *e = nil;

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), mil, wd, nil);
    if (!desc) { [pool drain]; return -1; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    if (!mdl) { [pool drain]; return -1; }

    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [blob writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    BOOL ok;
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) { NSLog(@"compile failed %d->%d: %@", ic, oc, e); [pool drain]; return -1; }
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) { NSLog(@"load failed %d->%d: %@", ic, oc, e); [pool drain]; return -1; }

    int result = setup_model(mdl, ic, oc, seq);
    [pool drain];
    return result;
}

// Load INT8 quantized conv model onto ANE
// w_int8: [oc, ic] int8 row-major, scale: [oc] fp32, offset: [oc] fp32
int ane_load_model_int8(int ic, int oc, int seq, const int8_t *w_int8, const float *scale_fp32, const float *offset_fp32) {
    if (!g_init || g_count >= MAX_MODELS) return -1;
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

    NSData *mil = [gen_conv_mil_int8(ic, oc, seq) dataUsingEncoding:NSUTF8StringEncoding];
    
    // Build weight blobs: INT8 data, FP16 scale, FP16 offset
    NSData *blob_data = build_blob_int8(w_int8, oc * ic);
    
    _Float16 *scale_fp16 = (_Float16*)calloc(oc, sizeof(_Float16));
    _Float16 *offset_fp16 = (_Float16*)calloc(oc, sizeof(_Float16));
    for (int i = 0; i < oc; i++) {
        scale_fp16[i] = (_Float16)scale_fp32[i];
        offset_fp16[i] = (_Float16)offset_fp32[i];
    }
    NSData *blob_scale = build_blob_fp16(scale_fp16, oc);
    NSData *blob_offset = build_blob_fp16(offset_fp16, oc);
    free(scale_fp16);
    free(offset_fp16);
    
    NSDictionary *wd = @{
        @"@model_path/weights/weight_data.bin": @{@"offset":@0, @"data":blob_data},
        @"@model_path/weights/weight_scale.bin": @{@"offset":@0, @"data":blob_scale},
        @"@model_path/weights/weight_offset.bin": @{@"offset":@0, @"data":blob_offset}
    };
    NSError *e = nil;

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), mil, wd, nil);
    if (!desc) { NSLog(@"INT8 desc failed %d->%d", ic, oc); [pool drain]; return -1; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    if (!mdl) { NSLog(@"INT8 model failed %d->%d", ic, oc); [pool drain]; return -1; }

    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [blob_data writeToFile:[td stringByAppendingPathComponent:@"weights/weight_data.bin"] atomically:YES];
    [blob_scale writeToFile:[td stringByAppendingPathComponent:@"weights/weight_scale.bin"] atomically:YES];
    [blob_offset writeToFile:[td stringByAppendingPathComponent:@"weights/weight_offset.bin"] atomically:YES];

    BOOL ok;
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) { NSLog(@"INT8 compile failed %d->%d: %@", ic, oc, e); [pool drain]; return -1; }
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) { NSLog(@"INT8 load failed %d->%d: %@", ic, oc, e); [pool drain]; return -1; }

    int result = setup_model(mdl, ic, oc, seq);
    [pool drain];
    return result;
}

int ane_eval(int handle) {
    if (handle < 0 || handle >= g_count) return -1;
    NSError *e = nil;
    return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        g_models[handle].model, @selector(evaluateWithQoS:options:request:error:),
        21, @{}, g_requests[handle], &e) ? 0 : -2;
}

int ane_run(int handle, const float *input, float *output) {
    if (handle < 0 || handle >= g_count) return -1;
    ANEModel *m = &g_models[handle];
    IOSurfaceRef ioI = g_surfaces[m->ioInIdx].surface;
    IOSurfaceRef ioO = g_surfaces[m->ioOutIdx].surface;
    IOSurfaceLock(ioI, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(ioI), input, (size_t)m->ic * m->seq * sizeof(float));
    IOSurfaceUnlock(ioI, 0, NULL);
    int rc = ane_eval(handle);
    if (rc != 0) return rc;
    IOSurfaceLock(ioO, kIOSurfaceLockReadOnly, NULL);
    memcpy(output, IOSurfaceGetBaseAddress(ioO), (size_t)m->oc * m->seq * sizeof(float));
    IOSurfaceUnlock(ioO, kIOSurfaceLockReadOnly, NULL);
    return 0;
}

int ane_run_async(int handle, const float *input) {
    if (handle < 0 || handle >= g_count) return -1;
    ANEModel *m = &g_models[handle];
    IOSurfaceRef ioI = g_surfaces[m->ioInIdx].surface;
    IOSurfaceLock(ioI, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(ioI), input, (size_t)m->ic * m->seq * sizeof(float));
    IOSurfaceUnlock(ioI, 0, NULL);
    dispatch_group_async(g_ane_group, g_ane_queue, ^{
        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            g_models[handle].model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, g_requests[handle], &e);
    });
    return 0;
}

int ane_wait_read(int handle, float *output) {
    if (handle < 0 || handle >= g_count) return -1;
    dispatch_group_wait(g_ane_group, DISPATCH_TIME_FOREVER);
    ANEModel *m = &g_models[handle];
    IOSurfaceRef ioO = g_surfaces[m->ioOutIdx].surface;
    IOSurfaceLock(ioO, kIOSurfaceLockReadOnly, NULL);
    memcpy(output, IOSurfaceGetBaseAddress(ioO), (size_t)m->oc * m->seq * sizeof(float));
    IOSurfaceUnlock(ioO, kIOSurfaceLockReadOnly, NULL);
    return 0;
}

void ane_wait(void) { dispatch_group_wait(g_ane_group, DISPATCH_TIME_FOREVER); }
int ane_model_count(void) { return g_count; }
int ane_surface_count(void) { return g_surfCount; }

void *ane_get_input_ptr(int handle) {
    if (handle < 0 || handle >= g_count) return NULL;
    return IOSurfaceGetBaseAddress(g_surfaces[g_models[handle].ioInIdx].surface);
}
void *ane_get_output_ptr(int handle) {
    if (handle < 0 || handle >= g_count) return NULL;
    return IOSurfaceGetBaseAddress(g_surfaces[g_models[handle].ioOutIdx].surface);
}
int ane_get_ic(int h)  { return (h>=0&&h<g_count)?g_models[h].ic:-1; }
int ane_get_oc(int h)  { return (h>=0&&h<g_count)?g_models[h].oc:-1; }
int ane_get_seq(int h) { return (h>=0&&h<g_count)?g_models[h].seq:-1; }

// Write input directly into IOSurface (caller provides [IC, SEQ] C-contiguous data)
void ane_write_input(int handle, const float *data) {
    if (handle < 0 || handle >= g_count) return;
    ANEModel *m = &g_models[handle];
    IOSurfaceRef ioI = g_surfaces[m->ioInIdx].surface;
    IOSurfaceLock(ioI, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(ioI), data, (size_t)m->ic * m->seq * sizeof(float));
    IOSurfaceUnlock(ioI, 0, NULL);
}

// Read output directly from IOSurface into [OC, SEQ] C-contiguous buffer
void ane_read_output(int handle, float *data) {
    if (handle < 0 || handle >= g_count) return;
    ANEModel *m = &g_models[handle];
    IOSurfaceRef ioO = g_surfaces[m->ioOutIdx].surface;
    IOSurfaceLock(ioO, kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(ioO), (size_t)m->oc * m->seq * sizeof(float));
    IOSurfaceUnlock(ioO, kIOSurfaceLockReadOnly, NULL);
}

// Tile transpose: src[rows][cols] → dst[cols][rows], cache-friendly 16×16 blocks
static void tile_transpose(const float * __restrict__ src, float * __restrict__ dst, int rows, int cols) {
    const int T = 16;
    for (int r = 0; r < rows; r += T) {
        int rend = r + T < rows ? r + T : rows;
        for (int c = 0; c < cols; c += T) {
            int cend = c + T < cols ? c + T : cols;
            for (int ri = r; ri < rend; ri++)
                for (int ci = c; ci < cend; ci++)
                    dst[ci * rows + ri] = src[ri * cols + ci];
        }
    }
}

// ane_run_T: input [SEQ, IC] row-major, output [SEQ, OC] row-major
// Tile-transpose input to [IC, SEQ] for IOSurface, eval, transpose output back
int ane_run_T(int handle, const float *input_seq_ic, float *output_seq_oc) {
    if (handle < 0 || handle >= g_count) return -1;
    ANEModel *m = &g_models[handle];
    IOSurfaceRef ioI = g_surfaces[m->ioInIdx].surface;
    IOSurfaceRef ioO = g_surfaces[m->ioOutIdx].surface;
    IOSurfaceLock(ioI, 0, NULL);
    tile_transpose(input_seq_ic, (float*)IOSurfaceGetBaseAddress(ioI), m->seq, m->ic);
    IOSurfaceUnlock(ioI, 0, NULL);
    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        m->model, @selector(evaluateWithQoS:options:request:error:),
        21, @{}, g_requests[handle], &e);
    if (!ok) return -2;
    IOSurfaceLock(ioO, kIOSurfaceLockReadOnly, NULL);
    tile_transpose((const float*)IOSurfaceGetBaseAddress(ioO), output_seq_oc, m->oc, m->seq);
    IOSurfaceUnlock(ioO, kIOSurfaceLockReadOnly, NULL);
    return 0;
}

// ane_eval_direct: write input, eval, read output — single call, minimal overhead
int ane_eval_direct(int handle, const float *input, float *output) {
    if (handle < 0 || handle >= g_count) return -1;
    ANEModel *m = &g_models[handle];
    IOSurfaceRef ioI = g_surfaces[m->ioInIdx].surface;
    IOSurfaceRef ioO = g_surfaces[m->ioOutIdx].surface;
    // Write input
    IOSurfaceLock(ioI, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(ioI), input, (size_t)m->ic * m->seq * sizeof(float));
    IOSurfaceUnlock(ioI, 0, NULL);
    // Eval
    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        m->model, @selector(evaluateWithQoS:options:request:error:),
        21, @{}, g_requests[handle], &e);
    if (!ok) return -2;
    // Read output
    IOSurfaceLock(ioO, kIOSurfaceLockReadOnly, NULL);
    memcpy(output, IOSurfaceGetBaseAddress(ioO), (size_t)m->oc * m->seq * sizeof(float));
    IOSurfaceUnlock(ioO, kIOSurfaceLockReadOnly, NULL);
    return 0;
}

// ane_run_rowmajor: input [L, IC] row-major, output [L, OC] row-major
// Uses vDSP_mtrans for fast transpose (Apple Accelerate)
// When L==seq: pure vDSP path (fastest). L<seq: vDSP + scatter.
int ane_run_rowmajor(int handle, const float *input_rm, int L, float *output_rm) {
    if (handle < 0 || handle >= g_count) return -1;
    ANEModel *m = &g_models[handle];
    if (L > m->seq) return -3;
    IOSurfaceRef ioI = g_surfaces[m->ioInIdx].surface;
    IOSurfaceRef ioO = g_surfaces[m->ioOutIdx].surface;

    // === Transpose input [L, IC] -> IOSurface [IC, seq] ===
    IOSurfaceLock(ioI, 0, NULL);
    float *ioBuf = (float*)IOSurfaceGetBaseAddress(ioI);
    if (L == m->seq) {
        // Fast path: vDSP_mtrans [L][IC] -> [IC][L=seq]
        vDSP_mtrans(input_rm, 1, ioBuf, 1, m->ic, L);
    } else {
        // L < seq: transpose to temp, then scatter with zero-pad
        // For each channel ch: ioBuf[ch*seq+0..L-1] = transposed, [L..seq-1] = 0
        // Use vDSP_mtrans to a contiguous temp, then scatter
        float *tmp = (float*)malloc((size_t)m->ic * L * sizeof(float));
        vDSP_mtrans(input_rm, 1, tmp, 1, m->ic, L);
        for (int ch = 0; ch < m->ic; ch++) {
            memcpy(&ioBuf[ch * m->seq], &tmp[ch * L], L * sizeof(float));
            if (L < m->seq)
                memset(&ioBuf[ch * m->seq + L], 0, (m->seq - L) * sizeof(float));
        }
        free(tmp);
    }
    IOSurfaceUnlock(ioI, 0, NULL);

    // === ANE eval ===
    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        m->model, @selector(evaluateWithQoS:options:request:error:),
        21, @{}, g_requests[handle], &e);
    if (!ok) return -2;

    // === Transpose output IOSurface [OC, seq] -> [L, OC] ===
    IOSurfaceLock(ioO, kIOSurfaceLockReadOnly, NULL);
    const float *outBuf = (const float*)IOSurfaceGetBaseAddress(ioO);
    if (L == m->seq) {
        // Fast path: vDSP_mtrans [OC][seq] -> [seq][OC]
        vDSP_mtrans(outBuf, 1, output_rm, 1, L, m->oc);
    } else {
        // Gather first L elements from each of OC rows, then transpose
        float *tmp = (float*)malloc((size_t)m->oc * L * sizeof(float));
        for (int ch = 0; ch < m->oc; ch++)
            memcpy(&tmp[ch * L], &outBuf[ch * m->seq], L * sizeof(float));
        vDSP_mtrans(tmp, 1, output_rm, 1, L, m->oc);
        free(tmp);
    }
    IOSurfaceUnlock(ioO, kIOSurfaceLockReadOnly, NULL);
    return 0;
}
