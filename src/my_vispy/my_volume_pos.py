from vispy.visuals.volume import VolumeVisual
from vispy.visuals.shaders import Function
from vispy.visuals import Visual

_VERTEX_SHADER = """
attribute vec3 a_position;
uniform vec3 u_shape;

varying vec3 v_position;
varying vec4 v_nearpos;
varying vec4 v_farpos;

void main() {
    v_position = a_position;

    // Project local vertex coordinate to camera position. Then do a step
    // backward (in cam coords) and project back. Voila, we get our ray vector.
    vec4 pos_in_cam = $viewtransformf(vec4(v_position, 1));

    // intersection of ray and near clipping plane (z = -1 in clip coords)
    pos_in_cam.z = -pos_in_cam.w;
    v_nearpos = $viewtransformi(pos_in_cam);

    // intersection of ray and far clipping plane (z = +1 in clip coords)
    pos_in_cam.z = pos_in_cam.w;
    v_farpos = $viewtransformi(pos_in_cam);

    gl_Position = $transform(vec4(v_position, 1.0));
}
"""  # noqa

_FRAGMENT_SHADER1 = """
// uniforms
uniform $sampler_type u_volumetex;
uniform vec3 u_shape;
uniform vec2 clim;
uniform float gamma;
uniform float u_threshold;
uniform float u_attenuation;
uniform float u_relative_step_size;
uniform float u_mip_cutoff;
uniform float u_minip_cutoff;

//varyings
varying vec3 v_position;
varying vec4 v_nearpos;
varying vec4 v_farpos;

// uniforms for lighting. Hard coded until we figure out how to do lights
const vec4 u_ambient = vec4(0.2, 0.2, 0.2, 1.0);
const vec4 u_diffuse = vec4(0.8, 0.2, 0.2, 1.0);
const vec4 u_specular = vec4(1.0, 1.0, 1.0, 1.0);
const float u_shininess = 40.0;

// uniforms for plane definition. Defined in data coordinates.
uniform vec3 u_plane_normal;
uniform vec3 u_plane_position;
uniform float u_plane_thickness;

//varying vec3 lightDirs[1];

// global holding view direction in local coordinates
vec3 view_ray;

float rand(vec2 co)
{
    // Create a pseudo-random number between 0 and 1.
    // http://stackoverflow.com/questions/4200224
    return fract(sin(dot(co.xy ,vec2(12.9898, 78.233))) * 43758.5453);
}

float colorToVal(vec4 color1)
{
    return color1.r; // todo: why did I have this abstraction in visvis?
}

vec4 applyColormap(float data) {
    data = clamp(data, min(clim.x, clim.y), max(clim.x, clim.y));
    data = (data - clim.x) / (clim.y - clim.x);
    vec4 color = $cmap(pow(data, gamma));
    return color;
}


vec4 calculateColor(vec4 betterColor, vec3 loc, vec3 step)
{   
    // Calculate color by incorporating lighting
    vec4 color1;
    vec4 color2;

    // View direction
    vec3 V = normalize(view_ray);

    // calculate normal vector from gradient
    vec3 N; // normal
    color1 = $get_data(loc+vec3(-step[0],0.0,0.0) );
    color2 = $get_data(loc+vec3(step[0],0.0,0.0) );
    N[0] = colorToVal(color1) - colorToVal(color2);
    betterColor = max(max(color1, color2),betterColor);
    color1 = $get_data(loc+vec3(0.0,-step[1],0.0) );
    color2 = $get_data(loc+vec3(0.0,step[1],0.0) );
    N[1] = colorToVal(color1) - colorToVal(color2);
    betterColor = max(max(color1, color2),betterColor);
    color1 = $get_data(loc+vec3(0.0,0.0,-step[2]) );
    color2 = $get_data(loc+vec3(0.0,0.0,step[2]) );
    N[2] = colorToVal(color1) - colorToVal(color2);
    betterColor = max(max(color1, color2),betterColor);
    float gm = length(N); // gradient magnitude
    N = normalize(N);

    // Flip normal so it points towards viewer
    float Nselect = float(dot(N,V) > 0.0);
    N = (2.0*Nselect - 1.0) * N;  // ==  Nselect * N - (1.0-Nselect)*N;

    // Get color of the texture (albeido)
    color1 = betterColor;
    color2 = color1;
    // todo: parametrise color1_to_color2

    // Init colors
    vec4 ambient_color = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 diffuse_color = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 specular_color = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 final_color;

    // todo: allow multiple light, define lights on viewvox or subscene
    int nlights = 1; 
    for (int i=0; i<nlights; i++)
    { 
        // Get light direction (make sure to prevent zero devision)
        vec3 L = normalize(view_ray);  //lightDirs[i]; 
        float lightEnabled = float( length(L) > 0.0 );
        L = normalize(L+(1.0-lightEnabled));

        // Calculate lighting properties
        float lambertTerm = clamp( dot(N,L), 0.0, 1.0 );
        vec3 H = normalize(L+V); // Halfway vector
        float specularTerm = pow( max(dot(H,N),0.0), u_shininess);

        // Calculate mask
        float mask1 = lightEnabled;

        // Calculate colors
        ambient_color +=  mask1 * u_ambient;  // * gl_LightSource[i].ambient;
        diffuse_color +=  mask1 * lambertTerm;
        specular_color += mask1 * specularTerm * u_specular;
    }

    // Calculate final color by componing different components
    final_color = color2 * ( ambient_color + diffuse_color) + specular_color;
    final_color.a = color2.a;

    // Done
    return final_color;
}


vec3 intersectLinePlane(vec3 linePosition, 
                        vec3 lineVector, 
                        vec3 planePosition, 
                        vec3 planeNormal) {
    // function to find the intersection between a line and a plane
    // line is defined by position and vector
    // plane is defined by position and normal vector
    // https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection

    // find scale factor for line vector
    float scaleFactor = dot(planePosition - linePosition, planeNormal) / 
                        dot(lineVector, planeNormal);

    // calculate intersection
    return linePosition + ( scaleFactor * lineVector );
}

// for some reason, this has to be the last function in order for the
// filters to be inserted in the correct place...

void main() {
    vec3 farpos = v_farpos.xyz / v_farpos.w;
    vec3 nearpos = v_nearpos.xyz / v_nearpos.w;

    // Calculate unit vector pointing in the view direction through this
    // fragment.
    view_ray = normalize(farpos.xyz - nearpos.xyz);

    // Variables to keep track of where to set the frag depth.
    // frag_depth_point is in data coordinates.
    vec3 frag_depth_point;

    // Set up the ray casting
    // This snippet must define three variables:
    // vec3 start_loc - the starting location of the ray in texture coordinates
    // vec3 step - the step vector in texture coordinates
    // int nsteps - the number of steps to make through the texture

    $raycasting_setup

    // For testing: show the number of steps. This helps to establish
    // whether the rays are correctly oriented
    //gl_FragColor = vec4(0.0, f_nsteps / 3.0 / u_shape.x, 1.0, 1.0);
    //return;

    $before_loop

    // This outer loop seems necessary on some systems for large
    // datasets. Ugly, but it works ...
    vec3 loc = start_loc;
    int iter = 0;

    // keep track if the texture is ever sampled; if not, fragment will be discarded
    // this allows us to discard fragments that only traverse clipped parts of the texture
    bool texture_sampled = false;

    while (iter < nsteps) {
        for (iter=iter; iter<nsteps; iter++)
        {
            // Only sample volume if loc is not clipped by clipping planes
            float distance_from_clip = $clip_with_planes(loc, u_shape);
            if (distance_from_clip >= 0)
            {
                // Get sample color
                vec4 color = $get_data(loc);
                float val = color.r;
                texture_sampled = true;

                $in_loop
            }
            // Advance location deeper into the volume
            loc += step;
        }
    }

    if (!texture_sampled)
        discard;

    $after_loop

    // set frag depth
    vec4 frag_depth_vector = vec4(frag_depth_point, 1);
    vec4 iproj = $viewtransformf(frag_depth_vector);
    iproj.z /= iproj.w;
    gl_FragDepth = (iproj.z+1.0)/2.0;
}
"""  # noqa

_shaders = {
    'vertex': _VERTEX_SHADER,
    'fragment': _FRAGMENT_SHADER1,
}

_MIP_SNIPPETS = dict(
    before_loop="""
        float maxval = u_mip_cutoff; // The maximum encountered value
        int maxi = -1;  // Where the maximum value was encountered
        """,
    in_loop="""
        if ( val > maxval ) {
            maxval = val;
            maxi = iter;
            if ( maxval >= clim.y ) {
                // stop if no chance of finding a higher maxval
                iter = nsteps;
            }
        }
        """,
    after_loop="""
        // Refine search for max value, but only if anything was found
        if ( maxi > -1 ) {
            // Calculate starting location of ray for sampling
            vec3 start_loc_refine = start_loc + step * (float(maxi) - 0.5);
            loc = start_loc_refine;

            // Variables to keep track of current value and where max was encountered
            vec3 max_loc_tex = start_loc_refine;

            vec3 small_step = step * 0.1;
            for (int i=0; i<10; i++) {
                float val = $get_data(loc).r;
                if ( val > maxval) {
                    maxval = val;
                    max_loc_tex = start_loc_refine + (small_step * i);
                }
                loc += small_step;
            }
            frag_depth_point = max_loc_tex * u_shape;
            
            gl_FragColor = vec4(max_loc_tex.z, max_loc_tex.y, max_loc_tex.x, 1);
            // gl_FragColor = applyColormap(maxval);
        } else {
            discard;
        }
        """,

)
# gl_FragColor = vec4(applyColormap(maxval)[0], max_loc_tex.x, max_loc_tex.y, max_loc_tex.z);
class MyVolPosVisual(VolumeVisual):
    def __init__(self, vol, clim="auto", method='mip', threshold=None,
                 attenuation=1.0, relative_step_size=0.8, cmap='grays',
                 gamma=1.0, interpolation='linear', texture_format=None,
                 raycasting_mode='volume', plane_position=None,
                 plane_normal=None, plane_thickness=1.0, clipping_planes=None,
                 clipping_planes_coord_system='scene', mip_cutoff=None,
                 minip_cutoff=None):

        super().__init__(vol=vol, clim=clim, method=method, threshold=threshold,
                         attenuation=attenuation, relative_step_size=relative_step_size,
                         cmap=cmap, gamma=gamma, interpolation=interpolation,
                         texture_format=texture_format, raycasting_mode=raycasting_mode,
                         plane_position=plane_position, plane_normal=plane_normal,
                         plane_thickness=plane_thickness, clipping_planes=clipping_planes,
                         clipping_planes_coord_system=clipping_planes_coord_system,
                         mip_cutoff=mip_cutoff, minip_cutoff=minip_cutoff)

        self._rendering_methods = {
            'mip': _MIP_SNIPPETS
            # Add other methods as needed
        }
        self.method = method
        self.shared_program.vert = _shaders['vertex']  # Vertex shader
        self.shared_program.frag = _shaders['fragment']  # Fragment shader
        # self.set_data(vol, clim or "auto", False)

    @property
    def method(self):
        """The render method to use

        Current options are:

            * translucent: voxel colors are blended along the view ray until
              the result is opaque.
            * mip: maxiumum intensity projection. Cast a ray and display the
              maximum value that was encountered.
            * minip: minimum intensity projection. Cast a ray and display the
              minimum value that was encountered.
            * attenuated_mip: attenuated maximum intensity projection. Cast a
              ray and display the maximum value encountered. Values are
              attenuated as the ray moves deeper into the volume.
            * additive: voxel colors are added along the view ray until
              the result is saturated.
            * iso: isosurface. Cast a ray until a certain threshold is
              encountered. At that location, lighning calculations are
              performed to give the visual appearance of a surface.
            * average: average intensity projection. Cast a ray and display the
              average of values that were encountered.
        """
        # print('frag: ', self.shared_program.frag)
        return self._method

    @method.setter
    def method(self, method):
        # Check and save
        known_methods = list(self._rendering_methods.keys())
        if method not in known_methods:
            raise ValueError('Volume render method should be in %r, not %r' %
                             (known_methods, method))
        self._method = method

        # $get_data needs to be unset and re-set, since it's present inside the snippets.
        #       Program should probably be able to do this automatically
        self.shared_program.frag['get_data'] = None
        self.shared_program.frag['raycasting_setup'] = self._raycasting_setup_snippet
        self.shared_program.frag['before_loop'] = self._before_loop_snippet
        self.shared_program.frag['in_loop'] = self._in_loop_snippet
        self.shared_program.frag['after_loop'] = self._after_loop_snippet
        self.shared_program.frag['sampler_type'] = self._texture.glsl_sampler_type
        self.shared_program.frag['cmap'] = Function(self._cmap.glsl_map)
        self.shared_program['texture2D_LUT'] = self.cmap.texture_lut()
        self.shared_program['u_mip_cutoff'] = self._mip_cutoff
        self.shared_program['u_minip_cutoff'] = self._minip_cutoff
        self._need_interpolation_update = True
        self.update()

    @property
    def _raycasting_setup_snippet(self):
        return self._raycasting_modes[self.raycasting_mode]

    @property
    def _before_loop_snippet(self):
        return self._rendering_methods[self.method]['before_loop']

    @property
    def _in_loop_snippet(self):
        return self._rendering_methods[self.method]['in_loop']

    @property
    def _after_loop_snippet(self):
        return self._rendering_methods[self.method]['after_loop']
