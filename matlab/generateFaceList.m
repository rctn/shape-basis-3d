function displayList = generateFaceList(face, GL)

    displayList = glGenLists(1);
    glNewList(displayList, GL.COMPILE);  
    glPointSize(1);
    glBegin(GL.POINTS);   
    for i=1:size(face,1) 
        if (face(i, 1) ~= 0 || face(i, 2) ~= 0 || face(i, 3) ~= 0 || face(i, 4) ~= 0 || face(i, 5) ~= 0 || face(i, 6) ~= 0)
            
            r = double(face(i, 4)/255);
            g = double(face(i, 5)/255);
            b = double(face(i, 6)/255);
            
            glMaterialfv(GL.FRONT_AND_BACK, GL.AMBIENT, [r-0.5, g-0.5 , b-0.5, 1.0]);
            glMaterialfv(GL.FRONT_AND_BACK, GL.DIFFUSE, [r, g, b, 1.0]); 
            glMaterialfv(GL.FRONT_AND_BACK, GL.SPECULAR, [r, g, b, 1.0]); 
            glMaterialfv(GL.FRONT_AND_BACK, GL.SHININESS, (1.1-((r+g+b)/3))*50);
        
            glVertex3f(face(i, 1), face(i, 2) , face(i, 3)); 
            glNormal3f(face(i, 1), face(i, 2) , face(i, 3));
            
        end
    end
    glEnd();
    glEndList(); 

end