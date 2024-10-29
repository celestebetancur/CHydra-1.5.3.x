 // This code is pretty much a collection of GLSL helper functions from multiple sources
 // plus the hydra by Olivia Jack. WGSL version and some other spices added by Celeste BEtancur Gutierrez
 
 "
    // include chugl standard vertex shader and other uniforms used by rendering engine
    #include FRAME_UNIFORMS
    #include DRAW_UNIFORMS
    #include STANDARD_VERTEX_INPUT
    #include STANDARD_VERTEX_OUTPUT
    #include STANDARD_VERTEX_SHADER

    @group(1) @binding(0) var<uniform> u_Time : f32;

    // video stuff
    @group(1) @binding(1) var u_sampler : sampler;
    @group(1) @binding(2) var u_texture : texture_2d<f32>;

    const PI = 3.141592653589793;
    const HALF_PI = 1.5707963267948966;

    var<private> v_TexCoord : vec2f;

    fn linear(t : f32) -> f32{
        return t;
    }

    fn rand(x : f32, y : f32) -> f32{
        return fract(sin(dot(vec2f(x, y) ,vec2f(12.9898,78.233))) * 43758.5453);
    }

    fn sawtooth(t : f32) -> f32 {
        return t - floor(t);
    }

    fn sineIn(t : f32) -> f32 {
        return sin((t - 1.0) * HALF_PI) + 1.0;
    }

    fn sineOut(t : f32) -> f32 {
        return sin(t * HALF_PI);
    }

    fn sineInOut(t : f32) -> f32{
        return -0.5 * (cos(PI * t) - 1.0);
    }

    fn qinticIn(t : f32) -> f32{
        return pow(t, 5.0);
    }

    fn qinticOut(t : f32) -> f32{
        return 1.0 - (pow(t - 1.0, 5.0));
    }

    fn backIn(t : f32) -> f32{
        return pow(t, 3.0) - t * sin(t * PI);
    }

    fn backOut(t : f32) -> f32{
        let f = 1.0 - t;
        return 1.0 - (pow(f, 3.0) - f * sin(f * PI));
    }

    fn permute(x : vec4f) -> vec4f{ 
        return ((x*34.0)+1.0)* (x - (289.0 * floor(x / 289.0)));
    }   

    fn taylorInvSqrt(r : vec4f) -> vec4f{
        return 1.79284291400159 - 0.85373472095314 * r;
    }

    fn _luminance(rgb:vec3f)->f32{
        let W = vec3(0.2125, 0.7154, 0.0721);
        return dot(rgb, W);
    }

    fn _rgbToHsv(c:vec3f)->vec3f{
        let K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
        let p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
        let q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
        let d = q.x - min(q.w, q.y);
        let e = 1.0e-10;
        return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
    }

    //  TODO: check this function
    //  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    //  this is not equivalent, GLSL clamp() to what in WGSL ???

    fn _hsvToRgb(c:vec3f)->vec3f{
        let K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, p - K.xxx, c.y);
    }

    fn _noise(v : vec3f) -> f32{
        let C :vec2f = vec2f(1.0/6.0, 1.0/3.0) ;
        let  D = vec4f(0.0, 0.5, 1.0, 2.0);
        // First corner
        var i  = floor(v + dot(v, C.yyy) );
        let x0 =   v - i + dot(i, C.xxx) ;
        // Other corners
        let g = step(x0.yzx, x0.xyz);
        let l = 1.0 - g;
        let i1 = min( g.xyz, l.zxy );
        let i2 = max( g.xyz, l.zxy );
        //  x0 = x0 - 0. + 0.0 * C
        let x1 = x0 - i1 + 1.0 * C.xxx;
        let x2 = x0 - i2 + 2.0 * C.xxx;
        let x3 = x0 - 1. + 3.0 * C.xxx;
        // Permutations
        i = i - (289.0 * floor(i/289.0));
        let p = permute( permute( permute(
                                        i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
                                + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
                        + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
        // Gradients
        // ( N*N points uniformly over a square, mapped onto an octahedron.)
        let n_ = 1.0/7.0; // N=7
        let  ns = n_ * D.wyz - D.xzx;
        let j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)
        let x_ = floor(j * ns.z);
        let y_ = floor(j - 7.0 * x_ );    // mod(j,N)
        let x = x_ *ns.x + ns.yyyy;
        let y = y_ *ns.x + ns.yyyy;
        let h = 1.0 - abs(x) - abs(y);
        let b0 = vec4( x.xy, y.xy );
        let b1 = vec4( x.zw, y.zw );
        let s0 = floor(b0)*2.0 + 1.0;
        let s1 = floor(b1)*2.0 + 1.0;
        let sh = -step(h, vec4(0.0));
        let a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
        let a1 = b1.xzyw + s1.xzyw*sh.zzww ;
        var p0 = vec3(a0.xy,h.x);
        var p1 = vec3(a0.zw,h.y);
        var p2 = vec3(a1.xy,h.z);
        var p3 = vec3(a1.zw,h.w);
        //Normalise gradients
        let norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
        p0 *= norm.x;
        p1 *= norm.y;
        p2 *= norm.z;
        p3 *= norm.w;
        // Mix final noise value
        var m = max((vec4f(0.6f, 0.6f, 0.6f, 0.6f) - vec4f(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3))), vec4f(0.0f, 0.0f, 0.0f, 0.0f));
        m = m * m;
        return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                    dot(p2,x2), dot(p3,x3) ) );
    }

    // Hydra

    fn solid (r : f32, g : f32, b : f32, a : f32) -> vec4f {
        return vec4f(r, g, b, a);
    }

    fn shape(_st : vec2f, sides : f32, radius : f32, smoothing : f32) -> vec4f{
        var st = _st * 2. - 1.;
        let a = atan2(st.x,st.y) + PI;
        let r = (2.* PI)/sides;
        let d = cos(floor(.5+a/r)*r-a)*length(st);
        return vec4f(vec3f(1.0-smoothstep(radius,radius + smoothing,d)), 1.0);
    }

    fn gradient( _st : vec2f, speed : f32) -> vec4f {
        return vec4(_st, sin(u_Time*speed), 1.0);
    }

    fn osc(_st : vec2f, freq : f32, sync : f32, offset : f32) -> vec4f {
        let st = _st;
        let r = sin((st.x-offset*2.0f/freq+u_Time*sync)*freq)*0.5f  + 0.5f;
        let g = sin((st.x+u_Time*sync)*freq)*0.5f + 0.5f;
        let b = sin((st.x+offset/freq+u_Time*sync)*freq)*0.5f  + 0.5f;
        return vec4f(r, g, b, 1.0);
    }

     fn voronoi( st : vec2f, scale: f32,speed:f32, blending:f32) -> vec4f{
   		var _st = st;
        var color = vec3(0.0);
        // Scale
        _st *= scale;
        // Tile the space
        var i_st : vec2f = floor(_st);
        let f_st = fract(_st);
        var m_dist = 10.0;  // minimun distance
        var m_point = vec2f(0);        // minimum point
        for (var j :f32=-1; j<=1; j = j + 1 ) {
            for (var i :f32=-1; i<=1; i = i + 1 ) {
               var neighbor : vec2f = vec2f(i,j);
                var p = i_st + neighbor;
                var point = fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);
                point = 0.5 + 0.5*sin(u_Time*speed + 6.2831*point);
                let diff = neighbor + point - f_st;
                let dist = length(diff);
                if( dist < m_dist ) {
                    m_dist = dist;
                    m_point = point;
                }
            }
        }
        // Assign a color using the closest point position
        color += dot(m_point,vec2(.3,.6));
        color *= 1.0 - blending*m_dist;
        return vec4(color, 1.0);
    }

    fn noise(_st : vec2f, scale : f32, offset : f32) -> vec4f {
        return vec4f(vec3f(_noise(vec3f(_st*scale, offset*u_Time))), 1.0);
    }

    // TODO: textures as sampler create a WGSL error
    // fn src(_st:vec2f) -> vec4f {
    //     return textureSample(u_texture, u_sampler, _st)
    // }

    fn rotate( _st : vec2f, angle : f32, speed : f32) -> vec2f{
        var xy = _st - vec2f(0.5);
        let ang = angle + speed * u_Time;
        xy = mat2x2f(cos(ang),-sin(ang), sin(ang),cos(ang))*xy;
        xy += 0.5;
        return xy;
    }

    fn scale( _st : vec2f, amount : f32, xMult : f32, yMult : f32, offsetX : f32, offsetY : f32) -> vec2f{
        var xy = _st - vec2f(offsetX, offsetY);
        xy *= (1.0/vec2f(amount*xMult, amount*yMult));
        xy += vec2f(offsetX, offsetY);
        return xy;
    }

    fn pixelate ( _st:vec2f, pixelX:f32, pixelY:f32) -> vec2f{
        let xy = vec2(pixelX, pixelY);
        return (floor(_st * xy) + 0.5)/xy;
    }

    fn repeat(_st:vec2f, repeatX:f32, repeatY:f32, offsetX:f32, offsetY:f32) -> vec2f{
        var st = _st * vec2(repeatX, repeatY);
        st.x += step(1., (st.y - (2.0 * floor(st.y / 2.0)))) * offsetX;
        st.y += step(1., (st.x - (2.0 * floor(st.y / 2.0)))) * offsetY;
        return fract(st);
    }

    fn modulateRepeat(_st:vec2f, _c0:vec4f, repeatX:f32, repeatY:f32, offsetX:f32, offsetY:f32) -> vec2f {
        var st = _st * vec2(repeatX, repeatY);
        st.x += step(1., (st.y - (2.0 * floor(st.y / 2.0)))) + _c0.r * offsetX;
        st.y += step(1., (st.x - (2.0 * floor(st.y / 2.0)))) + _c0.g * offsetY;
        return fract(st);
    }

    fn repeatX ( _st:vec2f, reps:f32, offset:f32)->vec2f{
        var st = _st * vec2(1.0, reps);
        st.x += step(1., (st.x - (2.0 * floor(st.x / 2.0)))) * offset;
        return fract(st);
    }

    fn modulateRepeatX( _st:vec2f, _c0:vec4f, reps:f32, offset:f32)-> vec2f{
        var st = _st * vec2(reps,1.0);
        st.y += step(1.0, (st.y - (2.0 * floor(st.y / 2.0)))) + _c0.r * offset;
        return fract(st);
    }

    fn repeatY (_st:vec2f, reps:f32, offset:f32)->vec2f{
        var st = _st * vec2(reps, 1.0);
        st.y += step(1., (st.x - (2.0 * floor(st.x / 2.0)))) * offset;
        return fract(st);
    }

    fn modulateRepeatY(_st:vec2f, _c0:vec4f, reps:f32, offset:f32)->vec2f{
        var st = _st * vec2(reps,1.0);
        st.x += step(1.0, (st.x - (2.0 * floor(st.y / 2.0)))) + _c0.r * offset;
        return fract(st);
    }

    fn kaleid(_st:vec2f, nSides:f32) -> vec2f{
        var st = _st;
        st -= 0.5;
        let r = length(st);
        var a = atan2(st.y, st.x);
        let pi = 2. * PI;
        let k = pi/nSides;
        a = a - (k * floor(a / k));
        a = abs(a-pi/nSides/2.);
        return r*vec2(cos(a), sin(a));
    }

    fn modulateKaleid(_st:vec2f, _c0:vec4f, nSides:f32)->vec2f{
        var st = _st - 0.5;
        let r = length(st);
        var a = atan2(st.y, st.x);
        let pi = 2. * PI;
        let k = pi/nSides;
        a = a - (k * floor(a / k));
        a = abs(a-pi/nSides/2.);
        return (_c0.r+r)*vec2(cos(a), sin(a));
    }

    fn scroll(st:vec2f, scrollX:f32, scrollY:f32, speedX:f32, speedY:f32)-> vec2f{
      	var _st = st;
        _st.x += scrollX + u_Time*speedX;
        _st.y += scrollY + u_Time*speedY;
        return fract(_st);
    }

    fn modulateScroll (st:vec2f, _c0:vec4f, scrollX:f32, scrollY:f32, speedX:f32, speedY:f32)->vec2f{
  		var _st = st;
        _st.x += _c0.r*scrollX + u_Time*speedX;
        _st.y += _c0.r*scrollY + u_Time*speedY;
        return fract(_st);
    }

    fn scrollX (st:vec2f, scrollX:f32, speed:f32)->vec2f{
      	var _st = st;
        _st.x += scrollX + u_Time*speed;
        return fract(_st);
    }

    fn modulateScrollX (st:vec2f, _c0:vec4f, scrollX:f32, speed:f32)->vec2f{
  		var _st = st;
        _st.x += _c0.r*scrollX + u_Time*speed;
        return fract(_st);
    }

    fn scrollY ( st:vec2f, scrollY:f32, speed:f32)->vec2f{
      	var _st = st;
        _st.y += scrollY + u_Time*speed;
        return fract(_st);
    }

    fn modulateScrollY (st:vec2f, _c0:vec4f, scrollY:f32, speed:f32)->vec2f{
  		var _st = st;
        _st.y += _c0.r*scrollY + u_Time*speed;
        return fract(_st);
    }

    fn posterize( _c0:vec4f, bins:f32, gamma:f32)->vec4f{
        var c2 = pow(_c0, vec4(gamma));
        c2 *= vec4(bins);
        c2 = floor(c2);
        c2/= vec4(bins);
        c2 = pow(c2, vec4(1.0/gamma));
        return vec4(c2.xyz, _c0.a);
    }

    fn shift( _c0:vec4f, r:f32, g:f32, b:f32, a:f32)->vec4f{
        var c2 = vec4(_c0);
        c2.r += fract(r);
        c2.g += fract(g);
        c2.b += fract(b);
        c2.a += fract(a);
        return c2.rgba;
    }

    fn add(_c0:vec4f, _c1:vec4f, amount:f32)->vec4f{
        return (_c0+_c1)*amount + _c0*(1.0-amount);
    }

    fn sub(_c0:vec4f, _c1:vec4f, amount:f32)->vec4f{
        return (_c0-_c1)*amount + _c0*(1.0-amount);
    }

    fn layer(_c0:vec4f, _c1:vec4f)->vec4f{
        return vec4(mix(_c0.rgb, _c1.rgb, _c1.a), _c0.a+_c1.a);
    }

    fn blend( _c0:vec4f, _c1:vec4f, amount:f32)->vec4f{
        return _c0*(1.0-amount)+_c1*amount;
    }

    fn mult(_c0:vec4f, _c1:vec4f, amount:f32)->vec4f{
        return _c0*(1.0-amount)+(_c0*_c1)*amount;
    }

    fn diff(_c0:vec4f, _c1:vec4f, amount:f32)->vec4f{
        return vec4(abs(_c0.rgb-_c1.rgb), max(_c0.a, _c1.a));
    }

    fn modulate(_st:vec2f, _c0:vec4f, amount:f32)->vec2f{
        return _st + _c0.xy * amount;
    }

    fn modulateScale(_st:vec2f, _c0:vec4f, offset:f32, multiple:f32)->vec2f{
        var xy = _st - vec2(0.5);
        xy*=(1.0/vec2(offset + multiple*_c0.r, offset + multiple*_c0.g));
        xy+=vec2(0.5);
        return xy;
    }

    fn modulatePixelate(_st:vec2f, _c0:vec4f, offset:f32, multiple:f32)->vec2f{
        var xy = vec2(offset + _c0.x*multiple, offset + _c0.y*multiple);
        return (floor(_st * xy) + 0.5)/xy;
    }

    fn modulateRotate(_st:vec2f, _c0:vec4f, offset:f32, multiple:f32)->vec2f{
        var xy = _st - vec2(0.5);
        let angle = offset + _c0.x * multiple;
        xy = mat2x2(cos(angle),-sin(angle), sin(angle),cos(angle))*xy;
        xy += 0.5;
        return xy;
    }

    fn modulateHue( _st:vec2f, _c0:vec4f, amount:f32)->vec2f{
        let t = _st + vec2(_c0.g - _c0.r, _c0.b - _c0.g) * amount / v_TexCoord;
        return t;
    }

    fn invert(_c0:vec4f, amount:f32)->vec4f{
        return vec4((1.0-_c0.rgb)*amount + _c0.rgb*(1.0-amount), _c0.a);
    }

    fn contrast(_c0:vec4f, amount:f32)->vec4f{
        let c = (_c0-vec4(0.5))*vec4(amount) + vec4(0.5);
        return vec4(c.rgb, _c0.a);
    }

    fn brightness(_c0:vec4f, amount:f32)->vec4f{
        return vec4(_c0.rgb + vec3(amount), _c0.a);
    }

    fn mask(_c0:vec4f, _c1:vec4f)->vec4f{
        let a = _luminance(_c1.rgb);
        return vec4(_c0.rgb*a, a*_c0.a);
    }

    fn luma(_c0:vec4f, threshold:f32, tolerance:f32)->vec4f{
        let a = smoothstep(threshold-(tolerance+0.0000001), threshold+(tolerance+0.0000001), _luminance(_c0.rgb));
        return vec4(_c0.rgb*a, a);
    }

    fn thresh(_c0:vec4f, threshold:f32, tolerance:f32)->vec4f{
        return vec4(vec3(smoothstep(threshold-tolerance, threshold+tolerance, _luminance(_c0.rgb))), _c0.a);
    }

    fn color(_c0:vec4f, r:f32, g:f32, b:f32, a:f32)->vec4f{
        let c = vec4(r, g, b, a);
        let pos = step(vec4f(0), c); // detect whether negative
        return vec4(mix((1.0-_c0)*abs(c), c*_c0, pos));
    }

    fn r(_c0:vec4f, scale:f32, offset:f32)->vec4f{
        return vec4(_c0.r * scale + offset);
    }

    fn g(_c0:vec4f, scale:f32, offset:f32)->vec4f{
        return vec4(_c0.g * scale + offset);
    }

    fn b(_c0:vec4f, scale:f32, offset:f32)->vec4f{
        return vec4(_c0.b * scale + offset);
    }

    fn a(_c0:vec4f, scale:f32, offset:f32)->vec4f{
        return vec4(_c0.a * scale + offset);
    }

    fn saturate(_c0:vec4f, amount:f32)->vec4f{
        let W = vec3(0.2125,0.7154,0.0721);
        let intensity = vec3(dot(_c0.rgb,W));
        return vec4(mix(intensity,_c0.rgb,amount), _c0.a);
    }

    fn hue(_c0:vec4f, hue:f32)->vec4f{
        var c = _rgbToHsv(_c0.rgb);
        c.r += hue;
        return vec4(_hsvToRgb(c), _c0.a);
    }

    fn colorama(_c0:vec4f, amount:f32)->vec4f{
        var c = _rgbToHsv(_c0.rgb);
        c += vec3(amount);
        c = _hsvToRgb(c);
        c = fract(c);
        return vec4(c, _c0.a);
    }

    fn chroma(c0:vec4f)->vec4f{
  		var _c0 = c0;
        let maxrb = max( _c0.r, _c0.b );
        let k = clamp( (_c0.g-maxrb)*5.0, 0.0, 1.0 );
        let dg = _c0.g; 
        _c0.g = min( _c0.g, maxrb*0.8 ); 
        _c0 += vec4(dg - _c0.g);
        return vec4(_c0.rgb, 1.0 - k);
    }

    fn modulateSR(_st:vec2f, _c0:vec4f, multiple:f32, offset:f32, rotateMultiple:f32, rotateOffset:f32)->vec2f{
        var xy = _st - vec2(0.5);
        let angle = rotateOffset + _c0.z * rotateMultiple;
        xy = mat2x2(cos(angle),-sin(angle), sin(angle),cos(angle))*xy;
        xy *= (1.0/vec2(offset + multiple*_c0.r, offset + multiple*_c0.g));
        xy += vec2(0.5);
        return xy;
    }

    fn sphere(_st:vec2f, radius:f32, rot:f32)->vec2f{
        let pos = _st-0.5;
        var rpos = vec3(0.0, 0.0, -10.0);
        let rdir = normalize(vec3(pos * 3.0, 1.0));
        var d = 0.0;
        for(var i = 1; i < 16; i += 1){
            d = length(rpos) - radius;
            rpos += d * rdir;
            if (abs(d) < 0.001){break;}
        }
        return vec2(atan2(rpos.z, rpos.x)+rot, atan2(length(rpos.xz), rpos.y));
    }

    fn sphereDisplacement(_st:vec2f, _c0:vec4f, radius:f32, rot:f32)->vec2f{
        var pos = _st-0.5;
        var rpos = vec3(0.0, 0.0, -10.0);
        let rdir = normalize(vec3(pos * 3.0, 1.0));
        var d = 0.0;
        for(var i = 1; i < 16; i += 1){
            let height = length(_c0);
            d = length(rpos) - (radius+height);
            rpos += d * rdir;
            if (abs(d) < 0.001){break;}
        }
        return vec2(atan2(rpos.z, rpos.x)+rot, atan2(length(rpos.xz), rpos.y));
    }

    fn modulateRays( _st:vec2f , _c0:vec4f, _scale:f32, samples:i32)->vec4f{
  		var scale = _scale;
        var uv = _st;
        var col = vec3(0.0);
        for (var i = 0; i < samples; i++) {
            scale -= 0.0002;
            uv -= 0.5;
            uv *= scale;
            uv += 0.5;
            col += smoothstep(vec3f(0.0), vec3f(1.0), _c0.rgb * 0.08);
        }

        return vec4(col,1.0);
    }

    fn srgbToLinear(c : vec4f) -> vec4f {
        return vec4f(
            pow(c.r, 2.2),
            pow(c.g, 2.2),
            pow(c.b, 2.2),
            c.a
        );
    }

    @fragment 
    fn fs_main(in : VertexOutput) -> @location(0) vec4f 
    {
        var uv : vec2f = in.v_uv;
        v_TexCoord = uv;
        // var texSampler = textureSample(u_texture, u_sampler, uv);
        var time = u_Time;

        //var FragColor = textureSample(u_texture, u_sampler, uv)
        var FragColor = " => global string hydra;

