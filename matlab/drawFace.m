function drawFace(face, face_num)
%%  INPUT PARAMETERS
%
%       face  = face data in one of the following formats:
%
%               n by 6 - [x,y,z,r,g,b]     seperate column vectors
%
%               n by 1 - [x]               single column vector
%                        [y]
%                        [z]
%                        [r]
%                        [g]
%                        [b]
%
%               n by m - [x][x][x][x]...   multiple column vectors
%                        [y][y][y][y]...
%                        [z][z][z][z]...
%                        [r][r][r][r]...
%                        [g][g][g][g]...
%                        [b][b][b][b]...
%
%               where:
%
%                   n = number of vertices
%                   m = number of faces
%
%       num_face = if full column matrix is given, the column (face) that
%       will be drawn
%
%%  CONTROLS
%       left mouse button  = rotate camera
%       right mouse button = rotate light
%       left/right key     = previous/next face (full column matrix only)
%
%%
    % convert from column format if given
    if (size(face,2) == 1), face = reshape(face, [], 6); end
    
    % given a full face matrix
    if (size(face,2) > 6)
        fullset = true;       
        matrix = face;
        if (nargin < 2), face_num = 1; end 
        face = reshape(face(:,face_num), [], 6);
    else
        fullset = false;
    end

    Screen('Preference', 'SkipSyncTests', 1);
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % All units are meant to be in millimeters.  glFrustum is set up that
    % the OpenGL coordinate system should be in millimeters.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % FRUSTUM PARAMETERS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % instantiate frustum
    LEFT    = -2400;
    RIGHT   = 2400;
    BOTTOM  = -1500;
    TOP     = 1500;  
    NEAR    = 10000;
    FAR     = 1000000;
    frustum = Frustum(LEFT, RIGHT, BOTTOM, TOP, NEAR, FAR);
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % WINDOW
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %set up window
    InitializeMatlabOpenGL;
    AssertOpenGL;
    windowPtr = initOpenGLNoLight(GL, 0);        
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % FACE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    FACE_SCALE = 8.0;
    displayList = generateFaceList(face, GL);
    
    % mouse coords
    MOUSE_SENSITIVITY = 0.4;
    old_x = 0;
    old_y = 0;
    
    % camera properties
    CAMERA_DISTANCE = 12000;
    camera = Frame([0,0,0], [0,0,-1], [0,1,0]);
    camera_rotation_amountX = 0;
    camera_rotation_amountY = 0;
    
    % light properties
    LIGHT_DISTANCE = 2000;
    light  = Frame([0, 0, 0], [0,0,-1], [0,1,0]);
    light_rotation_amountX = 0;
    light_rotation_amountY = 0;    
    light_ambient = single([0.5, 0.5, 0.5, 1.0]);
    light_diffuse = single([1.0, 1.0, 1.0, 1.0]);
    glLightfv(GL.LIGHT0, GL.AMBIENT, light_ambient);
    glLightfv(GL.LIGHT0, GL.DIFFUSE, light_diffuse);
       
    screenshot_count = 0;

    while(~getKey(27,0)) % exit on esc key press

        [x,y,buttons] = GetMouse();
        keysPressed = getKey([37, 39, 83], 0);
                
        % SCREENSHOT
        if (ismember(83, keysPressed))           
            imageArray = Screen('GetImage', windowPtr);
            imwrite(imageArray, strcat('ScreenShot', num2str(screenshot_count),'.png'));
            screenshot_count = screenshot_count +1;
        end
       
        % CHANGE FACE
        if (fullset == true)
            if (ismember(39, keysPressed))
                face_num = mod(face_num, size(matrix, 2)) + 1;
                glDeleteLists(displayList, 1);
                displayList = generateFaceList(reshape(matrix(:,face_num), [], 6), GL);               
            elseif (ismember(37, keysPressed))
                face_num = face_num - 1; if (face_num == 0), face_num = size(matrix,2); end  
                glDeleteLists(displayList, 1);
                displayList = generateFaceList(reshape(matrix(:,face_num), [], 6), GL);     
            end                
        end

        % update mouse position and delta
        if (old_x ~= 0 && old_y ~= 0)  
            if (buttons(1)) 
                camera_rotation_amountX = -(old_y - y)*MOUSE_SENSITIVITY;
                camera_rotation_amountY = (old_x - x)*MOUSE_SENSITIVITY; 
            elseif (length(buttons)>=2 && buttons(3))  
                light_rotation_amountX = -(old_y - y)*MOUSE_SENSITIVITY;
                light_rotation_amountY = (old_x - x)*MOUSE_SENSITIVITY;                               
            end           
        end
        old_x = x;
        old_y = y;

        Screen('BeginOpenGL',windowPtr);
        glClear();

        % set projection matrix
        glMatrixMode(GL.PROJECTION);
        glLoadIdentity();

        % set projection matrix
        frustum.applyProjectionMatrix();

        % set viewing matrix part of model view
        glMatrixMode(GL.MODELVIEW);
        glLoadIdentity();
        
        % LIGHTING POSITION
        light.setOrigin([0,0,0]);     
        if (length(buttons)>=3 && buttons(3))          
            % isolate world y and local x rotation to cause gimbol lock
            light.rotateWorld(-(light_rotation_amountY*pi)/180, [0,1,0]);
            light.rotateLocalX(-(light_rotation_amountX*pi)/180);           
        end        
        light.moveBackward(LIGHT_DISTANCE);
        glLightfv(GL.LIGHT0, GL.POSITION, [light.getOrigin(), 0]);
        
        if (length(buttons)>=3 && buttons(3))  
            % if light change is activated, show light position
            glPointSize(25);
            glBegin(GL.POINTS);
            glColor3ub(255,255,255);
            glVertex3f(light.getOriginX(), light.getOriginY(), light.getOriginZ() - CAMERA_DISTANCE);       
            glEnd();    
            glBegin (GL.LINES);
            glColor3ub(255,255,255);
            t = [0,0,0] - light.getOrigin();
            t = t/norm(t) * 500;
            glVertex3f(light.getOriginX(), light.getOriginY(), light.getOriginZ() - CAMERA_DISTANCE);  
            glVertex3f(light.getOriginX()+t(1), light.getOriginY() + t(2), light.getOriginZ() + t(3) - CAMERA_DISTANCE);               
            glEnd ();
        end
        
        % CAMERA POSITION
        % set camera on face, rotate into position, then zoom back out   
        camera.setOrigin([0,0,0]);
        if (buttons(1))         
            % isolate world y and local x rotation to cause gimbol lock
            camera.rotateWorld((camera_rotation_amountY*pi)/180, [0,1,0]);
            camera.rotateLocalX((camera_rotation_amountX*pi)/180);           
        end  
        camera.moveBackward(CAMERA_DISTANCE);
        camera.applyCameraMatrix()  
        
        % draw x,y,z axis
        glBegin (GL.LINES);
        glColor3ub(255,0,0);
        glVertex3f (0.0, 0.0, 0.0);
        glVertex3f (500.0, 0, 0.0);
        glEnd ();
        
        glBegin (GL.LINES);
        glColor3ub(0,255,0);
        glVertex3f (0.0, 0.0, 500.0);
        glVertex3f (0.0, 0.0, 0.0);
        glEnd ();
        
        glBegin (GL.LINES);
        glColor3ub(0,0,255);
        glVertex3f (0.0, 0.0, 0.0);
        glVertex3f (0.0, 500.0, 0.0);
        glEnd ();
                               
        % draw object   
        glEnable(GL.LIGHTING);
        glEnable(GL.LIGHT0);
        glScaled(FACE_SCALE, FACE_SCALE, FACE_SCALE);
        glCallList(displayList);
        glDisable(GL.LIGHTING);
        glDisable(GL.LIGHT0);
        glFlush();  
        
        Screen('EndOpenGL',windowPtr);
        Screen('Flip', windowPtr);
    end
    
    % close screen
    sca
    
end
    
