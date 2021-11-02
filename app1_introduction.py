import streamlit as st


def app():
    st.title("Actual Idiots")
    st.markdown('''*Driver Drowsiness Detection*''')

    # HTML Description
    page_body = '''
        <link rel="stylesheet" href="https://unicons.iconscout.com/release/v4.0.0/css/line.css">
        <h2 class="">About Us</h2>
            <p style="font-size:25px">
                Hey everyone we are team Actual Idiots! As part of the 7th Installment of the Data Associate Programme by SMU Business Intelligence and Analytics, we embarked on a project to determine the state of a person while he/she is driving. Special Thanks to our mentors Brian and Wa Thone who guided us over the last 12 weeks
            </p>

        <h2>Contributors</h2>
            <ul>
                <li style="font-size:20px">Sarah 
                    [<a href = "" target = "_blank">
                        LinkedIn
                    </a>]
                    </a>]
                    [<a href = "https://github.com/liam-ayathan" target = "_blank">
                        GitHub
                    </a>]  
                </li>
                <li style="font-size:20px">Maaruni
                    [<a href = "" target = "_blank">
                        LinkedIn
                    </a>]
                    </a>]
                    [<a href = "https://github.com/liam-ayathan" target = "_blank">
                        GitHub
                    </a>]    
                </li >
                <li style="font-size:20px">Joshua
                    [<a href = "" target = "_blank">
                        LinkedIn
                    </a>]
                    </a>]
                    [<a href = "https://github.com/liam-ayathan" target = "_blank">
                        GitHub
                    </a>]  
                </li>
                <li style="font-size:20px">Liam
                    [<a href = "https://www.linkedin.com/in/liam-ayathan-046b3816b/" target = "_blank">
                        LinkedIn
                    </a>]
                    </a>]
                    [<a href = "https://github.com/liam-ayathan" target = "_blank">
                        GitHub
                    </a>]  
                </li>
            </ul>
    '''
    st.markdown(page_body, unsafe_allow_html=True)
