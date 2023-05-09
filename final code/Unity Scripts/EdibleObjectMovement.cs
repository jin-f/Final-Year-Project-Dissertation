using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EdibleObjectMovement : MonoBehaviour
{
    public UDPReceive udpReceive;
    // Start is called before the first frame update
    float x = 0;
    float y = 0;
    float z = 0;
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        string received_data = udpReceive.coord_data;
        string[] coord_data = received_data.Split(',');

        if (coord_data[0] != "-inf" || coord_data[0] != "inf")
        {
            x = float.Parse(coord_data[0]) *10;
            y = (float.Parse(coord_data[1]) ) * 10;
            z = float.Parse(coord_data[2]) *10;
        }

        gameObject.transform.localPosition = new Vector3(x, y, z);


    }
}
