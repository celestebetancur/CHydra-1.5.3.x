@import "ChuGL.chug"
@import "Hydra.ck"

public class ShaderType {
    string code;
    string type;
}

public class CHydra {     

    Hydra hydra_object;
    hydra_object.hydra => string header;

    // TODO: make this modular
    "" => string textureInitializer;

    ";
    return srgbToLinear(FragColor);
    }
    " => string close;

    function ShaderType noise(){
        ShaderType toReturn;
        "noise ( uv , 10. , .1 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType noise(float freq){
        ShaderType toReturn;
        "noise ( uv , "+Std.ftoa(freq,10)+" , .1 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType noise(float freq, float offset){
        ShaderType toReturn;
        "noise ( uv , "+Std.ftoa(freq,10)+" , "+Std.ftoa(offset,10)+" )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType voronoi(){
        ShaderType toReturn;
        "voronoi ( uv , 100. , 1.0 , 0.3 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    } // scale speed blending

    function ShaderType voronoi(float scale){
        ShaderType toReturn;
        "voronoi ( uv , "+Std.ftoa(scale,10)+" , 1.0 , 0.3 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType voronoi(float scale, float speed){
        ShaderType toReturn;
        "voronoi ( uv , "+Std.ftoa(scale,10)+" , "+Std.ftoa(speed,10)+" , 0.3 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType voronoi(float scale, float speed, float blending){
        ShaderType toReturn;
        "voronoi ( uv , "+Std.ftoa(scale,10)+" , "+Std.ftoa(speed,10)+" , "+Std.ftoa(blending,10)+" )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType src(Webcam webcam){
        ShaderType toReturn;
        if(webcam.capture()){
            "src ( uv , u_texture)"=> toReturn.code;
            "src" => toReturn.type;
        }
        return toReturn;
    }

    function ShaderType src(Video video){
        ShaderType toReturn;
        "src ( uv , u_textureVideo)"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType osc(){
        ShaderType toReturn;
        "osc ( uv , 100. , 0.1 , 0.0 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType osc(float freq){
        ShaderType toReturn;
        "osc ( uv , "+Std.ftoa(freq,10)+" , 0.1 , 0.0 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }
    
    function ShaderType osc(float freq, float sync){
        ShaderType toReturn;
        "osc ( uv , "+Std.ftoa(freq,10)+" , "+Std.ftoa(sync,10)+",0.0 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType osc(float freq, float sync, float offset){
        ShaderType toReturn;
        "osc ( uv , "+Std.ftoa(freq,10)+" , "+Std.ftoa(sync,10)+" , "+Std.ftoa(offset,10)+" )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType shape(){
        ShaderType toReturn;
        "shape ( uv , 4. , 0.25 , 0.001 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType shape(float sides){
        ShaderType toReturn;
        "shape ( uv , "+Std.ftoa(sides,10)+" , 0.25 , 0.0 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType shape(float sides, float radius){
        ShaderType toReturn;
        "shape ( uv , "+Std.ftoa(sides,10)+" , "+Std.ftoa(radius,10)+" , 0.0 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType shape(float sides, float radius, float smoothing){
        ShaderType toReturn;
        "shape ( uv , "+Std.ftoa(sides,10)+" , "+Std.ftoa(radius,10)+" , "+Std.ftoa(smoothing,10)+" )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType gradient(){
        ShaderType toReturn;
        "gradient ( uv , 2.0 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType gradient(float speed){
        ShaderType toReturn;
        "gradient ( uv , "+Std.ftoa(speed,10)+" )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType solid(){
        ShaderType toReturn;
        "solid ( 0.0, 0.0, 0.0, 1.0)"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType solid(float r){
        ShaderType toReturn;
        "solid ( "+Std.ftoa(r,10)+", 0.0, 0.0, 1.0)"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType solid(float r, float g){
        ShaderType toReturn;
        "solid ( "+Std.ftoa(r,10)+", "+Std.ftoa(g,10)+", 0.0, 1.0)"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType solid(float r, float g, float b){
        ShaderType toReturn;
        "solid ( "+Std.ftoa(r,10)+", "+Std.ftoa(g,10)+", "+Std.ftoa(b,10)+", 1.0)"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType solid(float r, float g, float b, float a){
        ShaderType toReturn;
        "solid ( "+Std.ftoa(r,10)+", "+Std.ftoa(g,10)+", "+Std.ftoa(b,10)+", "+Std.ftoa(a,10)+")"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType rotate(){
        ShaderType toReturn;
        "rotate ( uv , .5 , .0 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType rotate(float angle){
        ShaderType toReturn;
        "rotate ( uv , "+Std.ftoa(angle,10)+" , 1. )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType rotate(float angle, float speed){
        ShaderType toReturn;
        "rotate ( uv , "+Std.ftoa(angle,10)+" , "+Std.ftoa(speed,10)+" )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scale(){
        ShaderType toReturn;
        "scale ( uv , 1.5 , 1.0 , 1.0 , 0.5 , 0.5)"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scale(float amount){
        ShaderType toReturn;
        "scale ( uv , "+Std.ftoa(amount,10)+" , 1.0 , 1.0 , 0.5 , 0.5)"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scale(float amount, float xMult){
        ShaderType toReturn;
        "scale ( uv , "+Std.ftoa(amount,10)+" , "+Std.ftoa(xMult,10)+" , 1.0 , 0.5 , 0.5)"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scale(float amount, float xMult, float yMult){
        ShaderType toReturn;
        "scale ( uv , "+Std.ftoa(amount,10)+" , "+Std.ftoa(xMult,10)+" , "+Std.ftoa(yMult,10)+" , 0.5 , 0.5)"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scale(float amount, float xMult, float yMult, float offsetX){
        ShaderType toReturn;
        "scale ( uv , "+Std.ftoa(amount,10)+" , "+Std.ftoa(xMult,10)+" , "+Std.ftoa(yMult,10)+" , "+Std.ftoa(offsetX,10)+" , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scale(float amount, float xMult, float yMult, float offsetX, float offsetY){
        ShaderType toReturn;
        "scale ( uv , "+Std.ftoa(amount,10)+" , "+Std.ftoa(xMult,10)+" , "+Std.ftoa(yMult,10)+" , "+Std.ftoa(offsetX,10)+" , "+Std.ftoa(offsetY,10)+" )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType pixelate(){
        ShaderType toReturn;
        "pixelate ( uv , 8. , 8. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType pixelate(float pixelX){
        ShaderType toReturn;
        "pixelate ( uv , "+Std.ftoa(pixelX,10)+" , 8. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType pixelate(float pixelX, float pixelY){
        ShaderType toReturn;
        "pixelate ( uv , "+Std.ftoa(pixelX,10)+" , "+Std.ftoa(pixelX,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatTex(){
        ShaderType toReturn;
        "repeat ( uv , 3. , 3. , 0. , 0. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatTex(float repeatX){
        ShaderType toReturn;
        "repeat ( uv , "+Std.ftoa(repeatX,10)+" , 3. , 0. , 0. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatTex(float repeatX, float repeatY){
        ShaderType toReturn;
        "repeat ( uv , "+Std.ftoa(repeatX,10)+" , "+Std.ftoa(repeatY,10)+" , 0. , 0. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatTex(float repeatX, float repeatY, float offsetX){
        ShaderType toReturn;
        "repeat ( uv , "+Std.ftoa(repeatX,10)+" , "+Std.ftoa(repeatY,10)+" , "+Std.ftoa(offsetX,10)+" , 0. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatTex(float repeatX, float repeatY, float offsetX, float offsetY){
        ShaderType toReturn;
        "repeat ( uv , "+Std.ftoa(repeatX,10)+" , "+Std.ftoa(repeatY,10)+" , "+Std.ftoa(offsetX,10)+" , "+Std.ftoa(offsetY,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

        function ShaderType modulateRepeat(ShaderType src){
        ShaderType toReturn;
        "modulateRepeat ( uv , "+src.code+" , 3. , 3. , 0.5 , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeat(ShaderType src, float repeatX){
        ShaderType toReturn;
        "modulateRepeat ( uv , "+src.code+" , "+Std.ftoa(repeatX,10)+" , 3. , 0.5 , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeat(ShaderType src, float repeatX, float repeatY){
        ShaderType toReturn;
        "modulateRepeat ( uv , "+src.code+" , "+Std.ftoa(repeatX,10)+" , "+Std.ftoa(repeatY,10)+" , 0.5 , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeat(ShaderType src, float repeatX, float repeatY, float offsetX){
        ShaderType toReturn;
        "modulateRepeat ( uv , "+src.code+" , "+Std.ftoa(repeatX,10)+" , "+Std.ftoa(repeatY,10)+" , "+Std.ftoa(offsetX,10)+" , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeat(ShaderType src, float repeatX, float repeatY, float offsetX, float offsetY){
        ShaderType toReturn;
        "modulateRepeat ( uv , "+src.code+" , "+Std.ftoa(repeatX,10)+" , "+Std.ftoa(repeatY,10)+" , "+Std.ftoa(offsetX,10)+" , "+Std.ftoa(offsetX,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatX(){
        ShaderType toReturn;
        "repeatX ( uv , 4. , 4. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatX(float reps){
        ShaderType toReturn;
        "repeatX ( uv , "+Std.ftoa(reps,10)+" , 4. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatX(float reps, float offset){
        ShaderType toReturn;
        "repeatX ( uv , "+Std.ftoa(reps,10)+" , "+Std.ftoa(offset,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeatX(ShaderType src){
        ShaderType toReturn;
        "modulateRepeatX ( uv , "+src.code+" , 3. , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeatX(ShaderType src, float reps){
        ShaderType toReturn;
        "modulateRepeatX ( uv , "+src.code+" , "+Std.ftoa(reps,10)+" , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeatX(ShaderType src, float reps, float offset){
        ShaderType toReturn;
        "modulateRepeatX ( tx , "+src.code+" , "+Std.ftoa(reps,10)+" , "+Std.ftoa(reps,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatY(){
        ShaderType toReturn;
        "repeatY ( uv , 3. , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatY(float reps){
        ShaderType toReturn;
        "repeatY ( uv , "+Std.ftoa(reps,10)+" , .5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatY(float reps, float offset){
        ShaderType toReturn;
        "repeatY ( uv , "+Std.ftoa(reps,10)+" , "+Std.ftoa(offset,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeatY(ShaderType src){
        ShaderType toReturn;
        "modulateRepeatY ( tx , "+src.code+" , 3. , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeatY(ShaderType src, float reps){
        ShaderType toReturn;
        "modulateRepeatY ( tx , "+src.code+" , "+Std.ftoa(reps,10)+" , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeatY(ShaderType src, float reps, float offset){
        ShaderType toReturn;
        "modulateRepeatY ( tx , "+src.code+" , "+Std.ftoa(reps,10)+" , "+Std.ftoa(reps,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType kaleid(){
        ShaderType toReturn;
        "kaleid ( uv , 4. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType kaleid(float nSides){
        ShaderType toReturn;
        "kaleid ( uv , "+Std.ftoa(nSides,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateKaleid(ShaderType src){
        ShaderType toReturn;
        "modulateKaleid ( uv , "+src.code+" , 4. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateKaleid(ShaderType src, float nSides){
        ShaderType toReturn;
        "modulateKaleid ( uv , "+src.code+" , "+Std.ftoa(nSides,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scroll(){
        ShaderType toReturn;
        "scroll ( uv , 5. , 5. , 0.5 , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scroll(float scrollX){
        ShaderType toReturn;
        "scroll ( uv , "+Std.ftoa(scrollX,10)+" , 5. , 0.5 , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scroll(float scrollX, float scrollY){
        ShaderType toReturn;
        "scroll ( uv , "+Std.ftoa(scrollX,10)+" , "+Std.ftoa(scrollY,10)+" , 0.5 , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scroll(float scrollX, float scrollY, float speedX){
        ShaderType toReturn;
        "scroll ( uv , "+Std.ftoa(scrollX,10)+" , "+Std.ftoa(scrollY,10)+" , "+Std.ftoa(speedX,10)+" , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scroll(float scrollX, float scrollY, float speedX, float speedY){
        ShaderType toReturn;
        "scroll ( uv , "+Std.ftoa(scrollX,10)+" , "+Std.ftoa(scrollY,10)+" , "+Std.ftoa(speedX,10)+" , "+Std.ftoa(speedY,10)+" )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scrollX(){
        ShaderType toReturn;
        "scrollX ( uv , 5. , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scrollX(float scrollX){
        ShaderType toReturn;
        "scrollX ( uv , "+Std.ftoa(scrollX,10)+" , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scrollX(float scrollX, float speed){
        ShaderType toReturn;
        "scrollX ( uv , "+Std.ftoa(scrollX,10)+" , "+Std.ftoa(speed,10)+" )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scrollY(){
        ShaderType toReturn;
        "scrollY ( uv , 5. , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scrollY(float scrollY){
        ShaderType toReturn;
        "scrollY ( uv , "+Std.ftoa(scrollY,10)+" , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scrollY(float scrollY, float speed){
        ShaderType toReturn;
        "scrollY ( uv , "+Std.ftoa(scrollY,10)+" , "+Std.ftoa(speed,10)+" )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScroll(ShaderType src){
        ShaderType toReturn;
        "modulateScroll ( uv , "+src.code+" , 0.5, 0.5 , 1.0 ,1.0 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScroll(ShaderType src ,float scrollX){
        ShaderType toReturn;
        "modulateScroll ( uv , "+src.code+" , "+Std.ftoa(scrollX,10)+", 0.5 , 1.0, 1.0 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScroll(ShaderType src ,float scrollX, float scrollY){
        ShaderType toReturn;
        "modulateScroll ( uv , "+src.code+" , "+Std.ftoa(scrollX,10)+", "+Std.ftoa(scrollY,10)+" , 1.0, 1.0 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScroll(ShaderType src ,float scrollX, float scrollY, float speedX){
        ShaderType toReturn;
        "modulateScroll ( uv , "+src.code+" , "+Std.ftoa(scrollX,10)+", "+Std.ftoa(scrollY,10)+" , "+Std.ftoa(speedX,10)+", 1.0 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScroll(ShaderType src ,float scrollX, float scrollY, float speedX, float speedY){
        ShaderType toReturn;
        "modulateScroll ( uv , "+src.code+" , "+Std.ftoa(scrollX,10)+", "+Std.ftoa(scrollY,10)+" , "+Std.ftoa(speedX,10)+", "+Std.ftoa(speedY,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScrollX(ShaderType src){
        ShaderType toReturn;
        "modulateScrollX ( uv , "+src.code+" , 0.5 , 1.0 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScrollX(ShaderType src, float scrollX){
        ShaderType toReturn;
        "modulateScrollX ( uv , "+src.code+" , "+Std.ftoa(scrollX,10)+" , 1.0 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScrollX(ShaderType src, float scrollX, float speed){
        ShaderType toReturn;
        "modulateScrollX ( uv , "+src.code+" , "+Std.ftoa(scrollX,10)+" , "+Std.ftoa(speed,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScrollY(ShaderType src){
        ShaderType toReturn;
        "modulateScrollY ( uv , "+src.code+" , 0.5 , 1.0 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScrollY(ShaderType src, float scrollY){
        ShaderType toReturn;
        "modulateScrollY ( uv , "+src.code+" , "+Std.ftoa(scrollY,10)+" , 1.0 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScrollY(ShaderType src, float scrollY, float speed){
        ShaderType toReturn;
        "modulateScrollY ( uv , "+src.code+" , "+Std.ftoa(scrollY,10)+" , "+Std.ftoa(speed,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType posterize(){
        ShaderType toReturn;
        "posterize ( tx , 3. , 0.6 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType posterize(float bins){
        ShaderType toReturn;
        "posterize ( tx , "+Std.ftoa(bins,10)+" , 0.6 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType posterize(float bins, float gamma){
        ShaderType toReturn;
        "posterize ( tx , "+Std.ftoa(bins,10)+" , "+Std.ftoa(gamma,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType shift(){
        ShaderType toReturn;
        "shift ( tx , 0.1 , 0.9 , 0.3 , 1.0)" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType shift(float r){
        ShaderType toReturn;
        "shift ( tx , "+Std.ftoa(r,10)+" , 0.9 , 0.3 , 1.0 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType shift(float r, float g){
        ShaderType toReturn;
        "shift ( tx , "+Std.ftoa(r,10)+" , "+Std.ftoa(g,10)+" , 0.3 , 1.0 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType shift(float r, float g, float b){
        ShaderType toReturn;
        "shift ( tx , "+Std.ftoa(r,10)+" , "+Std.ftoa(g,10)+" , "+Std.ftoa(b,10)+" , 1.0 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType shift(float r, float g, float b, float a){
        ShaderType toReturn;
        "shift ( tx , "+Std.ftoa(r,10)+" , "+Std.ftoa(g,10)+" , "+Std.ftoa(b,10)+" , "+Std.ftoa(a,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType add(ShaderType f1){
        ShaderType toReturn;
        "add ( tx , "+f1.code+" , 0.5 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType add(ShaderType f1, float amount){
        ShaderType toReturn;
        "add ( tx , "+f1.code+" , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType sub(ShaderType f1){
        ShaderType toReturn;
        "sub ( tx , "+f1.code+" , 0.5 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType sub(ShaderType f1, float amount){
        ShaderType toReturn;
        "sub ( tx , "+f1.code+" , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType layer(ShaderType f1){
        ShaderType toReturn;
        "layer ( tx , "+f1.code+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType blend(ShaderType f1){
        ShaderType toReturn;
        "blend ( tx , "+f1.code+" , 0.5 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType blend(ShaderType f1, float amount){
        ShaderType toReturn;
        "blend ( tx , "+f1.code+" , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType mult(ShaderType f1){
        ShaderType toReturn;
        "mult ( tx , "+f1.code+" , 0.5 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType mult(ShaderType f1, float amount){
        ShaderType toReturn;
        "mult ( tx , "+f1.code+" , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType diff(ShaderType f1){
        ShaderType toReturn;
        "diff ( tx , "+f1.code+" , 0.5 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType diff(ShaderType f1, float amount){
        ShaderType toReturn;
        "diff ( tx , "+f1.code+" , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulate(ShaderType src){
        ShaderType toReturn;
        "modulate ( uv , "+src.code+" , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScale(ShaderType src){
        ShaderType toReturn;
        "modulateScale ( uv , "+src.code+" , 1. , 1. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScale(ShaderType src, float offset){
        ShaderType toReturn;
        "modulateScale ( uv , "+src.code+" , "+Std.ftoa(offset,10)+ ", 1. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

   function ShaderType modulateScale(ShaderType src, float offset, float multiple){
        ShaderType toReturn;
        "modulateScale ( uv , "+src.code+" , "+Std.ftoa(offset,10)+ ", "+Std.ftoa(multiple,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulatePixelate(ShaderType src){
        ShaderType toReturn;
        "modulatePixelate ( uv , "+src.code+" , 3. , 10. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulatePixelate(ShaderType src, float offset){
        ShaderType toReturn;
        "modulatePixelate ( uv , "+src.code+" , "+Std.ftoa(offset,10)+ ", 10. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

   function ShaderType modulatePixelate(ShaderType src, float offset, float multiple){
        ShaderType toReturn;
        "modulatePixelate ( uv , "+src.code+" , "+Std.ftoa(offset,10)+ ", "+Std.ftoa(multiple,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRotate(ShaderType src){
        ShaderType toReturn;
        "modulateRotate ( uv , "+src.code+" , 1. , 1. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRotate(ShaderType src, float offset){
        ShaderType toReturn;
        "modulateRotate ( uv , "+src.code+" , "+Std.ftoa(offset,10)+ ", 1. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

   function ShaderType modulateRotate(ShaderType src, float offset, float multiple){
        ShaderType toReturn;
        "modulateRotate ( uv , "+src.code+" , "+Std.ftoa(offset,10)+ ", "+Std.ftoa(multiple,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateHue(ShaderType src){
        ShaderType toReturn;
        "modulateHue ( uv , "+src.code+" , 1. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateHue(ShaderType src, float ammount){
        ShaderType toReturn;
        "modulateHue ( uv , "+src.code+" , "+Std.ftoa(ammount,10)+ ")" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType invert(){
        ShaderType toReturn;
        "invert ( tx , 1. )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType invert(float amount){
        ShaderType toReturn;
        "invert ( tx , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType contrast(){
        ShaderType toReturn;
        "contrast ( tx , 1.6 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType contrast(float amount){
        ShaderType toReturn;
        "contrast ( tx , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType brightness(){
        ShaderType toReturn;
        "brightness ( tx , 0.4 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType brightness(float amount){
        ShaderType toReturn;
        "brightness ( tx , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType mask(ShaderType f1){
        ShaderType toReturn;
        "mask ( tx , "+f1.code+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType luma(){
        ShaderType toReturn;
        "luma ( tx , 0.5 , 0.1 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType luma(float threshold){
        ShaderType toReturn;
        "luma ( tx , "+Std.ftoa(threshold,10)+" , 0.1 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType luma(float threshold, float tolerance){
        ShaderType toReturn;
        "luma ( tx , "+Std.ftoa(threshold,10)+" , "+Std.ftoa(tolerance,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType thresh(){
        ShaderType toReturn;
        "thresh ( tx , 0.5 , 0.04 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType thresh(float threshold){
        ShaderType toReturn;
        "thresh ( tx , "+Std.ftoa(threshold,10)+" , 0.1 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType thresh(float threshold, float tolerance){
        ShaderType toReturn;
        "thresh ( tx , "+Std.ftoa(threshold,10)+" , "+Std.ftoa(tolerance,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType color(){
        ShaderType toReturn;
        "color ( tx , 1. , 1. , 1. , 1. )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType color(float r){
        ShaderType toReturn;
        "color ( tx , "+Std.ftoa(r,10)+" , 1. , 1. , 1. )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType color(float r, float g){
        ShaderType toReturn;
        "color ( tx , "+Std.ftoa(r,10)+" , "+Std.ftoa(g,10)+" , 1. , 1. )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType color(float r, float g, float b){
        ShaderType toReturn;
        "color ( tx , "+Std.ftoa(r,10)+" , "+Std.ftoa(g,10)+" , "+Std.ftoa(b,10)+" , 1. )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType color(float r, float g, float b, float a){
        ShaderType toReturn;
        "color ( tx , "+Std.ftoa(r,10)+" , "+Std.ftoa(g,10)+" , "+Std.ftoa(b,10)+" , "+Std.ftoa(a,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType r(){
        ShaderType toReturn;
        "r ( tx , 1. , 1.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType r(float scale){
        ShaderType toReturn;
        "r ( tx , "+Std.ftoa(scale,10)+" , 1.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType r(float scale, float offset){
        ShaderType toReturn;
        "r ( tx , "+Std.ftoa(scale,10)+" , "+Std.ftoa(offset,10)+"  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType g(){
        ShaderType toReturn;
        "g ( tx , 1. , 1.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType g(float scale){
        ShaderType toReturn;
        "g ( tx , "+Std.ftoa(scale,10)+" , 1.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType g(float scale, float offset){
        ShaderType toReturn;
        "g ( tx , "+Std.ftoa(scale,10)+" , "+Std.ftoa(offset,10)+"  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType b(){
        ShaderType toReturn;
        "b ( tx , 1. , 1.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType b(float scale){
        ShaderType toReturn;
        "b ( tx , "+Std.ftoa(scale,10)+" , 1.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType b(float scale, float offset){
        ShaderType toReturn;
        "b ( tx , "+Std.ftoa(scale,10)+" , "+Std.ftoa(offset,10)+"  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType a(){
        ShaderType toReturn;
        "a ( tx , 1. , 1.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType a(float scale){
        ShaderType toReturn;
        "a ( tx , "+Std.ftoa(scale,10)+" , 1.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType a(float scale, float offset){
        ShaderType toReturn;
        "a ( tx , "+Std.ftoa(scale,10)+" , "+Std.ftoa(offset,10)+"  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType saturate(){
        ShaderType toReturn;
        "saturate ( tx , 2.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType saturate(float amount){
        ShaderType toReturn;
        "saturate ( tx , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType hue(){
        ShaderType toReturn;
        "hue ( tx , 2.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType hue(float amount){
        ShaderType toReturn;
        "hue ( tx , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType colorama(){
        ShaderType toReturn;
        "colorama ( tx , 2.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType colorama(float amount){
        ShaderType toReturn;
        "colorama ( tx , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateSR(ShaderType src){
        ShaderType toReturn;
        "modulateSR ( uv , "+src.code+" , 4. , 0.5 , 2.0 , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateSR(ShaderType src, float multiple){
        ShaderType toReturn;
        "modulateSR ( uv , "+src.code+" , "+Std.ftoa(multiple,10)+" , 0.5 , 2.0 , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateSR(ShaderType src, float multiple, float offset){
        ShaderType toReturn;
        "modulateSR ( uv , "+src.code+" , "+Std.ftoa(multiple,10)+" , "+Std.ftoa(offset,10)+" , 2.0 , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateSR(ShaderType src, float multiple, float offset, float rotateMultiple){
        ShaderType toReturn;
        "modulateSR ( uv , "+src.code+" , "+Std.ftoa(multiple,10)+" , "+Std.ftoa(offset,10)+" , "+Std.ftoa(rotateMultiple,10)+" , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateSR(ShaderType src, float multiple, float offset, float rotateMultiple, float rotateOffset){
        ShaderType toReturn;
        "modulateSR ( uv , "+src.code+" , "+Std.ftoa(multiple,10)+" , "+Std.ftoa(offset,10)+" , "+Std.ftoa(rotateMultiple,10)+" , "+Std.ftoa(rotateOffset,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType chroma(){
        ShaderType toReturn;
        "chroma ( tx )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType sphere(){
        ShaderType toReturn;
        "sphere ( uv , 2. , 2. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType sphere(float radius){
        ShaderType toReturn;
        "sphere ( uv , "+Std.ftoa(radius,10)+" , 2. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType sphere(float radius, float rot){
        ShaderType toReturn;
        "sphere ( uv , "+Std.ftoa(radius,10)+" , "+Std.ftoa(rot,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType sphereDisplacement(ShaderType src){
        ShaderType toReturn;
        "sphereDisplacement ( uv , "+src.code+" , 2. , 2. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType sphereDisplacement(ShaderType src, float radius){
        ShaderType toReturn;
        "sphereDisplacement ( uv , "+src.code+" , "+Std.ftoa(radius,10)+" , 2. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType sphereDisplacement(ShaderType src, float radius, float rot){
        ShaderType toReturn;
        "sphereDisplacement ( uv , "+src.code+" , "+Std.ftoa(radius,10)+" , "+Std.ftoa(rot,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRays(){
        ShaderType toReturn;
        "modulateRays ( uv , tx , 1.0 , 100 )"=> toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRays(float scale){
        ShaderType toReturn;
        "modulateRays ( uv , tx , "+Std.ftoa(scale,10)+" , 100 )"=> toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRays(float scale, int samples){
        ShaderType toReturn;
        "modulateRays ( uv , tx , "+Std.ftoa(scale,10)+" , "+Std.itoa(samples)+" )"=> toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    //--------------------------------------
    function string shader(string code){
        return header+code+textureInitializer+close;
    }

    function string shader(ShaderType singleObject){
        return header+singleObject.code+textureInitializer+close;
    }

    function ShaderType CodeMaker(ShaderType one, ShaderType two, string type){

        StringTokenizer tokenizer;

        tokenizer.set( one.code );

        ShaderType toReturn;
        type => toReturn.type;

        while( tokenizer.more() )
        {
            tokenizer.next() => string temp;

            if(type == "src"){
                if(temp == "uv"){
                    two.code +=> toReturn.code;
                } else {
                    temp +=> toReturn.code; 
                }
                "src" => toReturn.type;
            }
            if(type == "srcC"){
                if(temp == "tx"){
                    two.code +=> toReturn.code;
                } else {
                    temp +=> toReturn.code; 
                }
                "src" => toReturn.type;
            }
        }
        
        return toReturn;
    }
}

//----------------------------------------------------------------
//  This section of the code will overload some operators

public ShaderType @operator ->(ShaderType one ,ShaderType two){
    CHydra st;
    if(one.type == "src" && two.type == "trns"){
        return st.CodeMaker(one, two, "src");
    }
    if(one.type == "src" && two.type == "trnsC"){
        return st.CodeMaker(two, one, "srcC");
    }

    return ShaderType n;
}

public ShaderType @operator +(ShaderType one ,ShaderType two){
    CHydra st;
    if(one.type == "src" && two.type == "src"){
        return one->st.add(two);
    }

    return ShaderType n;
}

public ShaderType @operator -(ShaderType one ,ShaderType two){
    CHydra st;
    if(one.type == "src" && two.type == "src"){
        return one->st.sub(two);
    }

    return ShaderType n;
}

public ShaderType @operator *(ShaderType one ,ShaderType two){
    CHydra st;
    if(one.type == "src" && two.type == "src"){
        return one->st.mult(two);
    }

    return ShaderType n;
}

public ShaderType @operator ^(ShaderType one ,ShaderType two){
    CHydra st;
    if(one.type == "src" && two.type == "src"){
        return one->st.diff(two);
    }

    return ShaderType n;
}

