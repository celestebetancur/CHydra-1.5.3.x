@import "CHydra"
CHydra st;
global string ShaderCode;
global int recompile;

(
    st.osc(130,1,1)
).code => ShaderCode;

1 => recompile;